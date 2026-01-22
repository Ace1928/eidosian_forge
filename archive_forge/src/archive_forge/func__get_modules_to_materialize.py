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
def _get_modules_to_materialize(root_module: nn.Module, ignored_modules: Set[nn.Module]) -> List[nn.Module]:
    modules_to_materialize: List[nn.Module] = []
    queue = collections.deque([root_module])
    visited_modules: Set[nn.Module] = {root_module}
    while queue:
        module = queue.popleft()
        modules_to_materialize.append(module)
        for child_module in module.children():
            if child_module not in visited_modules and _get_module_fsdp_state(child_module) is None and (child_module not in ignored_modules):
                visited_modules.add(child_module)
                queue.append(child_module)
    return modules_to_materialize