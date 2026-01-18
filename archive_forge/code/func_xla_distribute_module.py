import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
@with_xla
def xla_distribute_module(module: nn.Module, device_mesh: Optional[DeviceMesh]=None, partition_fn: Optional[Callable[[str, nn.Module, DeviceMesh], None]]=None, input_fn: Optional[Callable[..., None]]=None, output_fn: Optional[Callable[..., None]]=None) -> nn.Module:
    raise NotImplementedError