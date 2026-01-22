import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
class RecurrentGemmaRglru(nn.Module):
    """A Real-Gated Linear Recurrent Unit (RG-LRU) layer."""

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.block_width = config.lru_width // self.num_attention_heads
        self.recurrent_param = nn.Parameter(torch.empty([config.lru_width]))
        self.input_gate_weight = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width, self.block_width]))
        self.input_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))
        self.recurrent_gate_weight = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width, self.block_width]))
        self.recurrent_gate_bias = nn.Parameter(torch.empty([self.num_attention_heads, self.block_width]))
        self.recurrent_states = None

    def forward(self, activations: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, lru_width = activations.shape
        reset = position_ids[:, :, None] == 0
        reshape_act = activations.reshape(batch_size * seq_len, self.num_attention_heads, self.block_width)
        reshape_act = reshape_act.permute(1, 0, 2)
        res = torch.baddbmm(self.input_gate_bias[:, None, :], reshape_act, self.input_gate_weight)
        input_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))
        res = torch.baddbmm(self.recurrent_gate_bias[:, None, :], reshape_act, self.recurrent_gate_weight)
        recurrent_gate = torch.sigmoid(res.transpose(0, 1).reshape(batch_size, seq_len, lru_width))
        log_recurrent_gate = -8.0 * recurrent_gate * nn.functional.softplus(self.recurrent_param)
        recurrent_gate = torch.exp(log_recurrent_gate)
        a_square = torch.exp(2 * log_recurrent_gate)
        gated_inputs = activations * input_gate
        multiplier = 1
        tracing = isinstance(activations, torch.fx.Proxy) or (hasattr(torch, '_dynamo') and torch._dynamo.is_compiling())
        if not torch.jit.is_tracing() and (not tracing):
            multiplier = SqrtBoundDerivative.apply(1 - a_square)
        multiplier = reset + ~reset * multiplier
        normalized_x = gated_inputs * multiplier.type(activations.dtype)
        hidden_states, recurrent_states = self._rnn_scan(hidden_states=normalized_x, recurrent_gate=recurrent_gate, reset=reset, recurrent_states=self.recurrent_states)
        self.recurrent_states = recurrent_states
        return hidden_states

    def _rnn_scan(self, hidden_states: torch.Tensor, recurrent_gate: torch.Tensor, reset: torch.Tensor, recurrent_states: Union[torch.Tensor, None], acc_dtype: torch.dtype=torch.float32) -> Tuple[torch.Tensor, torch.Tensor]:
        """Runs the recurrence of a linear RNN.

        Args:
        hidden_states: The input sequence.
        recurrent_gate: The diagonal of the recurrence matrix `A`.
        reset: Indicator of document boundaries, e.g. when to reset the hidden state
            of the RNN.
        recurrent_states: The initial hidden state.
        acc_dtype: The data type for the accumulation.

        Returns:
        The output of the linear recurrence.
        """
        recurrent_gate = recurrent_gate * ~reset
        if hidden_states.shape[1] == 1:
            if recurrent_states is None:
                return (hidden_states, hidden_states[:, 0].type(acc_dtype))
            else:
                contextualized_states = recurrent_gate.type(acc_dtype) * recurrent_states[:, None].to(recurrent_gate.device)
                contextualized_states += hidden_states.type(acc_dtype)
                return (contextualized_states.type(hidden_states.dtype), contextualized_states[:, -1])
        else:
            if recurrent_states is None:
                recurrent_states = torch.zeros(hidden_states[:, 0].shape, dtype=acc_dtype, device=hidden_states.device)
            contextualized_states = torch.zeros_like(hidden_states)
            for t in range(hidden_states.shape[1]):
                recurrent_states = recurrent_gate[:, t].type(acc_dtype) * recurrent_states.to(recurrent_gate.device)
                recurrent_states = recurrent_states + hidden_states[:, t].type(acc_dtype)
                contextualized_states[:, t] = recurrent_states.type(hidden_states.dtype)
        return (contextualized_states, recurrent_states)