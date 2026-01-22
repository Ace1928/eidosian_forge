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
def _get_device_from_device_id(device_id: Optional[Union[int, torch.device]], rank: int) -> Optional[torch.device]:
    """
    Return a ``torch.device`` for the specified ``device_id``.

    Processes ``device_id`` and returns either the corresponding device or
    ``None`` if ``device_id`` is ``None``.
    """
    if device_id is None:
        return None
    device = device_id if isinstance(device_id, torch.device) else torch.device(device_id)
    if device == torch.device('cuda'):
        warnings.warn(f'FSDP got the argument `device_id` {device_id} on rank {rank}, which does not have an explicit index. FSDP will use the current device {torch.cuda.current_device()}. If this is incorrect, please explicitly call `torch.cuda.set_device()` before FSDP initialization or pass in the explicit device index as the `device_id` argument.')
        device = torch.device('cuda', torch.cuda.current_device())
    return device