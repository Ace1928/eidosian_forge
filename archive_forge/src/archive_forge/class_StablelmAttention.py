from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class StablelmAttention(nn.Module):

    def __init__(self, config: PretrainedConfig, linear_method: Optional[LinearMethodBase]=None) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_key_value_heads = config.num_key_value_heads
        if self.total_num_key_value_heads >= tp_size:
            assert self.total_num_key_value_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_key_value_heads == 0
        self.num_key_value_heads = max(1, self.total_num_key_value_heads // tp_size)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.max_position_embeddings = config.max_position_embeddings
        rope_pct = getattr(config, 'rope_pct', getattr(config, 'partial_rotary_factor', 1))
        self.rotary_ndims = int(self.head_dim * rope_pct)
        self.scaling = self.head_dim ** (-0.5)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim
        self.qkv_bias = getattr(config, 'use_qkv_bias', False)
        if self.head_dim * self.num_heads * tp_size != self.hidden_size:
            raise ValueError(f'hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`: {self.num_heads}).')
        self.qkv_proj = QKVParallelLinear(self.hidden_size, self.head_dim, self.total_num_heads, self.total_num_key_value_heads, self.qkv_bias, linear_method=linear_method)
        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim, self.hidden_size, bias=False, linear_method=linear_method)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.rotary_ndims, max_position=self.config.max_position_embeddings, base=self.config.rope_theta)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_key_value_heads)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.o_proj(attn_output)
        return output