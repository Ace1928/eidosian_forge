import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
from vllm.transformers_utils.configs.mpt import MPTConfig
class MPTForCausalLM(nn.Module):

    def __init__(self, config: MPTConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        assert config.tie_word_embeddings
        self.linear_method = linear_method
        self.transformer = MPTModel(config, linear_method)
        self.lm_head_weight = self.transformer.wte.weight
        self.sampler = Sampler(config.vocab_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision):
            if name.endswith('.bias') and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            weight_loader(param, loaded_weight)