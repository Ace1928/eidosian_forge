from typing import Optional
import torch
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,
class DeciLMForCausalLM(LlamaForCausalLM):
    """
    Implementation for https://huggingface.co/Deci/DeciLM-7b-instruct.
    Based on the llama executor.

    The main difference is that DeciLM uses Variable Grouped Query Attention.
    The constant number of GQA heads in the decoder is overridden with a value
    per layer.

    Usually, in the HuggingFace implementation, instead of
    "config.num_key_value_heads", we use
    "config.num_key_value_heads_per_layer[i]" which varies.

    Currently, PagedAttention does not work well with variable GQA, so we
    normalize the weights upon loading, and use uniform GQA with the max value
    instead.
    """

    def __init__(self, config: Optional[PretrainedConfig]=None, linear_method: Optional[LinearMethodBase]=None, lora_config: Optional[LoRAConfig]=None) -> None:
        config.num_key_value_heads = max(config.num_key_value_heads_per_layer)
        delattr(config, 'num_key_value_heads_per_layer')
        super().__init__(config=config, linear_method=linear_method, lora_config=lora_config)

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str]=None, load_format: str='auto', revision: Optional[str]=None):
        stacked_params_mapping = [('qkv_proj', 'q_proj', 'q'), ('qkv_proj', 'k_proj', 'k'), ('qkv_proj', 'v_proj', 'v'), ('gate_up_proj', 'gate_proj', 0), ('gate_up_proj', 'up_proj', 1)]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in hf_model_weights_iterator(model_name_or_path, cache_dir, load_format, revision):
            if 'rotary_emb.inv_freq' in name:
                continue
            if 'k_proj' in name or 'v_proj' in name:
                loaded_weight = self._degroup_weight(loaded_weight)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith('.bias') and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, 'weight_loader', default_weight_loader)
                weight_loader(param, loaded_weight)

    def _degroup_weight(self, loaded_weight: torch.Tensor) -> torch.Tensor:
        hidden_size = self.config.hidden_size
        head_size = self.config.hidden_size // self.config.num_attention_heads
        target_num_kv_heads = self.config.num_key_value_heads
        num_kv_heads = loaded_weight.shape[0] // head_size
        n_repeats = target_num_kv_heads / num_kv_heads
        assert n_repeats == int(n_repeats)
        n_repeats = int(n_repeats)
        loaded_weight = loaded_weight.view(num_kv_heads, head_size, hidden_size)
        loaded_weight = torch.repeat_interleave(loaded_weight, repeats=n_repeats, dim=0)
        loaded_weight = loaded_weight.reshape(target_num_kv_heads * head_size, hidden_size)
        return loaded_weight