import collections
import itertools
import os
import warnings
from typing import (
import torch
import torch.distributed as dist
import torch.distributed.fsdp._exec_order_utils as exec_order_utils
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.distributed.fsdp.fully_sharded_data_parallel as fsdp_file
import torch.nn as nn
from torch.distributed.algorithms._comm_hooks import default_hooks
from torch.distributed.device_mesh import _mesh_resources, DeviceMesh
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._common_utils import (
from torch.distributed.fsdp._flat_param import (
from torch.distributed.fsdp._limiter_utils import _FreeEventQueue
from torch.distributed.fsdp.api import (
from torch.distributed.fsdp.wrap import _Policy
from torch.distributed.tensor.parallel.fsdp import DTensorExtensions
from torch.distributed.utils import _sync_params_and_buffers
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils.hooks import RemovableHandle
def _get_ignored_modules(root_module: nn.Module, _ignored_modules: Optional[Iterable[torch.nn.Module]]) -> Set[nn.Module]:
    """
    Check that ``_ignored_modules`` is an iterable of ``nn.Module`` s without any FSDP instances.

    Return the modules contained in their module
    subtrees as a :class:`set`. Nested FSDP instances are excluded, but their
    already-computed ignored modules are included.

    ``_ignored_modules`` represents the argument passed by the user to FSDP.
    """
    msg_prefix = '`ignored_modules` should be an iterable of `torch.nn.Module`s '
    try:
        ignored_root_modules = set(_ignored_modules) if _ignored_modules is not None else set()
    except TypeError as e:
        raise TypeError(msg_prefix + f'but got {type(_ignored_modules)}') from e
    for module in ignored_root_modules:
        if not isinstance(module, torch.nn.Module):
            raise TypeError(msg_prefix + f'but got an iterable with {type(module)}')
        if _get_module_fsdp_state(module):
            raise ValueError('`ignored_modules` should not include FSDP modules')
    for module in root_module.modules():
        if not traversal_utils._composable(module):
            ignored_root_modules.add(module)
    ignored_modules = {child for module in ignored_root_modules for child in module.modules() if not isinstance(child, fsdp_file.FullyShardedDataParallel)}
    if root_module in ignored_modules:
        warnings.warn(f'Trying to ignore the top-level module passed into the FSDP constructor itself will result in all parameters being ignored and is not well-supported: {module}')
    for submodule in root_module.modules():
        optional_fsdp_state = _get_module_fsdp_state(submodule)
        if optional_fsdp_state is not None:
            assert hasattr(optional_fsdp_state, '_ignored_modules')
            ignored_modules.update(optional_fsdp_state._ignored_modules)
    return ignored_modules