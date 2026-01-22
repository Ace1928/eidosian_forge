import logging
import math
from enum import Enum
from typing import Callable
import torch
import torch.nn as nn
from torch.nn.init import (
def _init_weights_vit_timm(module: nn.Module, name: str='', gain: float=1.0, deepnorm_style: bool=False, **kwargs):
    """
    ViT weight initialization, original timm impl (for reproducibility).

    See DeepNet_ for all the DeepNorm specific codepaths
    """
    if isinstance(module, nn.Linear):
        if deepnorm_style and ('q_proj' in name.split('.') or 'k_proj' in name.split('.')):
            gain = 1
        std = 0.02 * gain
        a = math.sqrt(3.0) * std
        _maybe_init_tensor(module, 'weight', _no_grad_trunc_normal_, mean=0.0, std=std, a=-a, b=a)
        _maybe_init_tensor(module, 'bias', nn.init.zeros_)
    elif hasattr(module, 'init_weights'):
        module.init_weights(gain=gain)
    else:
        _maybe_report_no_init(module, name)
    if not hasattr(module, 'init_weights'):
        for child_name, child_module in module.named_children():
            _init_weights_vit_timm(child_module, child_name, gain)