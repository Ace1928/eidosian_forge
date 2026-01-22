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
def _infer_device_type(*args):
    device_types = list({arg.device.type for arg in args if isinstance(arg, torch.Tensor) and (not arg.device.type == 'cpu')})
    if len(device_types) > 1:
        warnings.warn('Tensor arguments, excluding CPU tensors, are detected on at least two types of devices. Device state will only be saved for devices of a single device type, and the remaining devices will be ignored. Consequently, if any checkpointed functions involve randomness, this may result in incorrect gradients. (Note that if CUDA devices are among the devices detected, it will be prioritized; otherwise, the first device encountered will be selected.)')
    if len(device_types) == 0:
        return DefaultDeviceType.get_device_type()
    elif 'cuda' in device_types:
        return 'cuda'
    else:
        return device_types[0]