import contextlib
import importlib
import os
import sys
import threading
import traceback
import warnings
from functools import lru_cache
from typing import Any, cast, List, Optional, Tuple, Union
import torch
import torch._C
from torch.types import Device
from .. import device as _device
from .._utils import classproperty
from ._utils import _dummy_type, _get_device_index
from .graphs import (
from .streams import Event, ExternalStream, Stream
from .memory import *  # noqa: F403
from .random import *  # noqa: F403
from torch.storage import _LegacyStorage, _warn_typed_storage_removal
from . import amp, jiterator, nvtx, profiler, sparse
def _check_cubins():
    incompatible_device_warn = '\n{} with CUDA capability sm_{} is not compatible with the current PyTorch installation.\nThe current PyTorch install supports CUDA capabilities {}.\nIf you want to use the {} GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/\n'
    if torch.version.cuda is None:
        return
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [int(arch.split('_')[1]) for arch in arch_list if 'sm_' in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        supported = any((sm // 10 == cap_major for sm in supported_sm))
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(incompatible_device_warn.format(device_name, capability, ' '.join(arch_list), device_name))