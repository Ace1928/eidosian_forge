import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_cohere import CohereConfig
@add_start_docstrings('The bare Cohere Model outputting raw hidden-states without any specific head on top.', COHERE_START_DOCSTRING)
class CoherePreTrainedModel(PreTrainedModel):
    config_class = CohereConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['CohereDecoderLayer']
    _skip_keys_device_placement = ['past_key_values']
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int]=None):
        if self.config._attn_implementation == 'flash_attention_2' and cache_cls == StaticCache:
            raise ValueError('`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers')
        for layer in self.model.layers:
            device = layer.input_layernorm.weight.device
            if hasattr(self.config, '_pre_quantization_dtype'):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            layer.self_attn.past_key_value = cache_cls(self.config, max_batch_size, max_cache_len, device=device, dtype=dtype)

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None