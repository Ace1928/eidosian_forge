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
class ChatGLMModel(nn.Module):

    def __init__(self, config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.embedding = VocabParallelEmbedding(config.padded_vocab_size, config.hidden_size)
        self.num_layers = config.num_layers
        self.multi_query_group_num = config.multi_query_group_num
        self.kv_channels = config.kv_channels
        self.encoder = GLMTransformer(config, linear_method)
        self.output_layer = ParallelLMHead(config.padded_vocab_size, config.hidden_size)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        inputs_embeds = self.embedding(input_ids)
        hidden_states = self.encoder(hidden_states=inputs_embeds, position_ids=position_ids, kv_caches=kv_caches, input_metadata=input_metadata)
        return hidden_states