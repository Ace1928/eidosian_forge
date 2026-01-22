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