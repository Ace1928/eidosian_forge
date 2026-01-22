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
def _get_root_modules(modules: Set[nn.Module]) -> Set[nn.Module]:
    """
    Returns:
        Set[nn.Module]: The subset of ``modules`` that are root modules (i.e.
        parent-less) with respect to the modules in the set itself. In other
        words, these are the modules in ``modules`` that are not the child of
        any other module in ``modules``.
    """
    root_modules: Set[nn.Module] = set()
    module_to_submodules = {module: set(module.modules()) for module in modules}
    for candidate_module in modules:
        is_root_module = True
        for module, submodules in module_to_submodules.items():
            is_child_module = candidate_module is not module and candidate_module in submodules
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.add(candidate_module)
    return root_modules