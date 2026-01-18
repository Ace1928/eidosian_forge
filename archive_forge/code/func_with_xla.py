import logging
import os
from functools import wraps
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from torch.distributed._tensor.placement_types import Placement, Replicate
from torch.distributed.device_mesh import DeviceMesh
def with_xla(func: Callable) -> Callable:
    assert func is not None

    @wraps(func)
    def wrapper(self, *args: Tuple[object], **kwargs: Dict[str, Any]) -> None:
        if TORCH_XLA_INITIALIZED:
            os.environ['XLA_USE_SPMD'] = '1'
            return func(self, *args, **kwargs)
        else:
            raise ImportError('torch.distributed._tensor._xla API requires torch_xla package installation.')
    return wrapper