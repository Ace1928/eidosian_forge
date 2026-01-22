from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
from vllm.sequence import SamplerOutput
from hf_olmo import OLMoConfig
class OlmoBlock(nn.Module):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: OLMoConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.attn = OlmoAttention(config, linear_method)
        self.mlp = OlmoMLP(config, linear_method)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        og_x = hidden_states
        x = self.attn(positions, hidden_states, kv_cache, input_metadata)
        x = x + og_x
        hidden_states = self.mlp(x)
        return hidden_states