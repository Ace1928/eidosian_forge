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
def _supports_autocast(device):
    device_module = _get_device_module(device)
    return device == 'cuda' or (hasattr(device_module, 'is_autocast_enabled') and hasattr(device_module, 'get_autocast_dtype'))