from typing import Optional
import torch
from transformers import PretrainedConfig
from vllm.config import LoRAConfig
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.weight_utils import (default_weight_loader,
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