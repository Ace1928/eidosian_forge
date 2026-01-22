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
@no_type_check
def _module_handle(state: _FSDPState, module: nn.Module) -> Optional['FlatParamHandle']:
    """
    Returns the ``FlatParamHandle`` s corresponding to ``module``. This is
    the handle that contains some parameter in ``module``.
    """
    if _is_composable(state):
        if state._handle is None:
            return None
        assert module in state._fully_sharded_module_to_handle, f'Expects a fully sharded module but got {module} on rank {state.rank}'
        return state._fully_sharded_module_to_handle[module]
    else:
        return module._handle