import logging
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, ContextManager, Literal, Mapping, Optional, Union
import torch
from lightning_utilities import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
from lightning_fabric.plugins.precision.precision import Precision
from lightning_fabric.plugins.precision.utils import (
from lightning_fabric.utilities.rank_zero import rank_zero_info, rank_zero_warn
def _convert_layers(module: torch.nn.Module) -> None:
    import transformer_engine.pytorch as te
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            if child.in_features % 8 != 0 or child.out_features % 16 != 0:
                rank_zero_warn(f'Support for FP8 in the linear layers with this plugin is currently limited to tensors with shapes where the dimensions are divisible by 8 and 16 respectively. The layer {name!r} does not fit this criteria. You might want to add padding to your inputs.')
                continue
            has_bias = child.bias is not None
            replacement = te.Linear(child.in_features, child.out_features, bias=has_bias)
            replacement.weight.data = child.weight.data.clone()
            if has_bias:
                replacement.bias.data = child.bias.data.clone()
            log.debug(f'Replacing layer {name!r} with Transformer Engine equivalent')
            module.__setattr__(name, replacement)
        elif isinstance(child, torch.nn.LayerNorm):
            replacement = te.LayerNorm(child.normalized_shape[0], eps=child.eps)
            replacement.weight.data = child.weight.data.clone()
            replacement.bias.data = child.bias.data.clone()
            log.debug(f'Replacing layer {name!r} with Transformer Engine equivalent')
            module.__setattr__(name, replacement)
        else:
            _convert_layers(child)