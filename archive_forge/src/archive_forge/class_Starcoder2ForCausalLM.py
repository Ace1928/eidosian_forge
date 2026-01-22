from typing import List, Optional, Tuple
import torch
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class Starcoder2ForCausalLM(nn.Module):

    def __init__(self, config: Starcoder2Config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.model = Starcoder2Model(config, linear_method=linear_method)
        self.vocab_size = config.vocab_size
        self.unpadded_vocab_size = config.vocab_size
        if config.tie_word_embeddings:
            self.lm_head_weight = self.model.embed_tokens.weight
        else:
            self.unpadded_vocab_size = config.vocab_size
            self.lm_head = ParallelLMHead(self.unpadded_vocab_size, config.hidden_size, org_num_embeddings=config.vocab_size, padding_size=DEFAULT_VOCAB_PADDING_SIZE)
            self.lm_head_weight = self.lm_head.weight
        self.sampler = Sampler(self.unpadded_vocab_size, config.vocab_size)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches, input_metadata)
        return hidden_states

    def sample(self, hidden_states: Optional[torch.Tensor], sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(self.lm_head_weight, hidden_states, sampling_metadata)
        return next_tokens

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v')]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision):
            if 'rotary_emb.inv_freq' in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if self.config.tie_word_embeddings and 'lm_head.weight' in name:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)