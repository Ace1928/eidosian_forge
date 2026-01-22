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
def _no_dispatch_record_stream(tensor: torch.Tensor, stream: torch.Stream) -> None:
    if tensor.device.type not in ['cuda', torch._C._get_privateuse1_backend_name()]:
        return
    if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        with no_dispatch():
            tensor.record_stream(stream)
    else:
        tensor.record_stream(stream)