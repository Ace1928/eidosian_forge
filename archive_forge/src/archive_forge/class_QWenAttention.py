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
class QWenAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int, max_position_embeddings: int, rope_theta: float=10000, rope_scaling: Optional[Dict[str, Any]]=None, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.hidden_size = hidden_size
        tensor_model_parallel_world_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tensor_model_parallel_world_size == 0
        self.num_heads = self.total_num_heads // tensor_model_parallel_world_size
        self.head_dim = hidden_size // self.total_num_heads
        self.c_attn = QKVParallelLinear(hidden_size, self.head_dim, self.total_num_heads, bias=True, linear_method=linear_method)
        self.c_proj = RowParallelLinear(self.total_num_heads * self.head_dim, hidden_size, bias=False, linear_method=linear_method)
        self.scaling = self.head_dim ** (-0.5)
        self.rotary_emb = get_rope(self.head_dim, rotary_dim=self.head_dim, max_position=max_position_embeddings, base=rope_theta, rope_scaling=rope_scaling)
        self.attn = PagedAttention(self.num_heads, self.head_dim, self.scaling)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        qkv, _ = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(chunks=3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        k_cache, v_cache = kv_cache
        attn_output = self.attn(q, k, v, k_cache, v_cache, input_metadata)
        output, _ = self.c_proj(attn_output)
        return output