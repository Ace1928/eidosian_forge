from __future__ import annotations
import math
import operator
import re
import warnings
from dataclasses import asdict, replace
from enum import Enum
from functools import reduce
from itertools import chain
from typing import Literal, Optional
import torch
from torch import nn
from tqdm import tqdm
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, check_target_module_exists, onload_layer
from peft.utils import (
from peft.utils.merge_utils import dare_linear, dare_ties, magnitude_prune, task_arithmetic, ties
from .aqlm import dispatch_aqlm
from .awq import dispatch_awq
from .config import LoraConfig
from .gptq import dispatch_gptq
from .layer import Conv2d, LoraLayer, dispatch_default
from .tp_layer import dispatch_megatron
def _create_and_replace(self, lora_config, adapter_name, target, target_name, parent, current_key):
    if current_key is None:
        raise ValueError("Current Key shouldn't be `None`")
    pattern_keys = list(chain(lora_config.rank_pattern.keys(), lora_config.alpha_pattern.keys()))
    target_name_key = next(filter(lambda key: re.match(f'.*\\.{key}$', current_key), pattern_keys), current_key)
    r = lora_config.rank_pattern.get(target_name_key, lora_config.r)
    alpha = lora_config.alpha_pattern.get(target_name_key, lora_config.lora_alpha)
    kwargs = {'r': r, 'lora_alpha': alpha, 'lora_dropout': lora_config.lora_dropout, 'fan_in_fan_out': lora_config.fan_in_fan_out, 'init_lora_weights': lora_config.init_lora_weights, 'use_rslora': lora_config.use_rslora, 'use_dora': lora_config.use_dora, 'loaded_in_8bit': getattr(self.model, 'is_loaded_in_8bit', False), 'loaded_in_4bit': getattr(self.model, 'is_loaded_in_4bit', False)}
    quant_methods = ['gptq', 'aqlm', 'awq']
    for quant_method in quant_methods:
        quantization_config = get_quantization_config(self.model, method=quant_method)
        if quantization_config is not None:
            kwargs[f'{quant_method}_quantization_config'] = quantization_config
    from peft.tuners.adalora import AdaLoraLayer
    if isinstance(target, LoraLayer) and (not isinstance(target, AdaLoraLayer)):
        target.update_layer(adapter_name, r, lora_alpha=alpha, lora_dropout=lora_config.lora_dropout, init_lora_weights=lora_config.init_lora_weights, use_rslora=lora_config.use_rslora, use_dora=lora_config.use_dora)
    else:
        new_module = self._create_new_module(lora_config, adapter_name, target, **kwargs)
        if adapter_name != self.active_adapter:
            new_module.requires_grad_(False)
        self._replace_module(parent, target_name, new_module, target)