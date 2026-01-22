from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import LayerNorm
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
from vllm.transformers_utils.configs import ChatGLMConfig
class GLMTransformer(nn.Module):
    """Transformer class."""

    def __init__(self, config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.post_layer_norm = config.post_layer_norm
        self.num_layers = config.num_layers
        self.layers = nn.ModuleList([GLMBlock(config, linear_method) for i in range(self.num_layers)])
        if self.post_layer_norm:
            layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
            self.final_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(hidden_states=hidden_states, position_ids=position_ids, kv_cache=kv_caches[i], input_metadata=input_metadata)
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states