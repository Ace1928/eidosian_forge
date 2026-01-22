import logging
import traceback
import warnings
import weakref
from enum import auto, Enum
from functools import partial
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._flat_param as flat_param_file
import torch.nn as nn
from torch.distributed._composable_state import _get_module_state, _State
from torch.distributed._tensor.device_mesh import DeviceMesh
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.utils import _apply_to_tensors
from torch.utils._mode_utils import no_dispatch
from .api import (
def _override_module_mixed_precision(root: torch.nn.Module, module_classes_to_override: Iterable[Type[nn.Module]], wrap_override_dict: Dict[str, Any]={'mixed_precision': None}) -> Set[Type[nn.Module]]:
    module_classes_to_override = tuple(set(module_classes_to_override))
    overridden_module_classes: Set[Type[nn.Module]] = set()
    for mod in root.modules():
        if isinstance(mod, module_classes_to_override):
            overridden_module_classes.add(type(mod))
            mod._wrap_overrides = wrap_override_dict

            def cast_fn(dtype: torch.dtype, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
                if not torch.is_floating_point(x) or x.dtype == dtype:
                    return x
                _MODULE_TO_INP_DTYPE[module] = x.dtype
                return x.to(dtype)

            def forward_pre_hook(module, args):
                return _apply_to_tensors(partial(cast_fn, torch.float32, module), args)

            def forward_post_hook(module, args, output):
                if module in _MODULE_TO_INP_DTYPE:
                    old_dtype = _MODULE_TO_INP_DTYPE[module]
                    return _apply_to_tensors(partial(cast_fn, old_dtype, module), output)
            mod.register_forward_pre_hook(forward_pre_hook, prepend=False)
            mod.register_forward_hook(forward_post_hook, prepend=False)
    return overridden_module_classes