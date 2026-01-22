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
class ChatGLMForCausalLM(nn.Module):

    def __init__(self, config: ChatGLMConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config: ChatGLMConfig = config
        self.linear_method = linear_method
        self.transformer = ChatGLMModel(config, linear_method)
        self.lm_head_weight = self.transformer.output_layer.weight
        self.sampler = Sampler(config.padded_vocab_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.transformer(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision):
            if 'rotary_pos_emb.inv_freq' in name:
                continue
            if 'word_embeddings' in name:
                name = name.replace('.word_embeddings', '')
            if name.endswith('.bias') and name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, 'weight_loader', default_weight_loader)
            weight_loader(param, loaded_weight)