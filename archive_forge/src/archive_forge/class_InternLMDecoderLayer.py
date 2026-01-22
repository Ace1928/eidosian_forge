from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class InternLMDecoderLayer(nn.Module):

    def __init__(self, config: PretrainedConfig, linear_method: Optional[LinearMethodBase]=None) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, 'rope_theta', 10000)
        rope_scaling = getattr(config, 'rope_scaling', None)
        max_position_embeddings = getattr(config, 'max_position_embeddings', 8192)
        self.attention = InternLM2Attention(hidden_size=self.hidden_size, num_heads=config.num_attention_heads, num_kv_heads=config.num_key_value_heads, rope_theta=rope_theta, rope_scaling=rope_scaling, max_position_embeddings=max_position_embeddings, linear_method=linear_method)
        self.feed_forward = InternLM2MLP(hidden_size=self.hidden_size, intermediate_size=config.intermediate_size, hidden_act=config.hidden_act, linear_method=linear_method)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, residual: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_norm(hidden_states)
        else:
            hidden_states, residual = self.attention_norm(hidden_states, residual)
        hidden_states = self.attention(positions=positions, hidden_states=hidden_states, kv_cache=kv_cache, input_metadata=input_metadata)
        hidden_states, residual = self.ffn_norm(hidden_states, residual)
        hidden_states = self.feed_forward(hidden_states)
        return (hidden_states, residual)