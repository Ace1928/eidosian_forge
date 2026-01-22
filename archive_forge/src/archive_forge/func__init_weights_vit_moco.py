import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _init_weights_vit_moco(module: nn.Module, name: str='', gain: float=1.0, **kwargs):
    """ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed"""
    assert 'deepnorm_style' not in kwargs.keys(), 'This initialization method does not support deepnorm'
    if is_ffn(name):
        _maybe_init_tensor(module, 'weight', torch.nn.init.xavier_uniform_, gain=gain)
        _maybe_init_tensor(module, 'bias', nn.init.zeros_)
    elif is_mha_input_projection(name) or isinstance(module, nn.Linear):
        if isinstance(module.weight, torch.Tensor):
            val = math.sqrt(6.0 / float(module.weight.shape[0] + module.weight.shape[1])) * gain
            _maybe_init_tensor(module, 'weight', nn.init.uniform_, a=-val, b=val)
        _maybe_init_tensor(module, 'bias', nn.init.zeros_)
    elif hasattr(module, 'init_weights'):
        module.init_weights(gain=gain)
    else:
        _maybe_report_no_init(module, name)
    if not hasattr(module, 'init_weights'):
        for child_name, child_module in module.named_children():
            _init_weights_vit_moco(child_module, child_name, gain)