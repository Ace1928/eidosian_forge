from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.mistral.configuration_mistral import MistralConfig
class UnpaddedMistralDecoderLayer(nn.Module):

    def __init__(self, config: MistralConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = UnpaddedMistralAttention(config=config)
        self.mlp = UnpaddedMistralMLP(config=config)
        self.input_layernorm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UnpaddedMistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, cos_sin: Tuple[torch.Tensor, torch.Tensor], nz_hidden_states: torch.Tensor, nz_position_ids: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        residual = nz_hidden_states
        nz_hidden_states = self.input_layernorm(nz_hidden_states)
        nz_hidden_states = self.self_attn(cos_sin=cos_sin, nz_hidden_states=nz_hidden_states, nz_position_ids=nz_position_ids, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
        nz_hidden_states = residual + nz_hidden_states
        residual = nz_hidden_states
        nz_hidden_states = self.post_attention_layernorm(nz_hidden_states)
        nz_hidden_states = self.mlp(nz_hidden_states)
        nz_hidden_states = residual + nz_hidden_states
        return nz_hidden_states