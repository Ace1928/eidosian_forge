import contextlib
import platform
import uuid
import warnings
import weakref
from collections import defaultdict
from itertools import count
from typing import (
from weakref import ReferenceType
import torch
import torch.fx.traceback as fx_traceback
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import capture_logs, LoggingTensorMode
from torch.utils._python_dispatch import TorchDispatchMode
def _get_autocast_kwargs(device='cuda'):
    if device == 'cuda':
        device_autocast_kwargs = {'enabled': torch.is_autocast_enabled(), 'dtype': torch.get_autocast_gpu_dtype(), 'cache_enabled': torch.is_autocast_cache_enabled()}
    elif _supports_autocast(device):
        device_module = _get_device_module(device)
        device_autocast_kwargs = {'enabled': device_module.is_autocast_enabled(), 'dtype': device_module.get_autocast_dtype(), 'cache_enabled': torch.is_autocast_cache_enabled()}
    else:
        device_autocast_kwargs = None
    cpu_autocast_kwargs = {'enabled': torch.is_autocast_cpu_enabled(), 'dtype': torch.get_autocast_cpu_dtype(), 'cache_enabled': torch.is_autocast_cache_enabled()}
    return (device_autocast_kwargs, cpu_autocast_kwargs)