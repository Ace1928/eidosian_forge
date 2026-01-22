import collections
import contextlib
import ctypes
import pickle
import sys
import warnings
from inspect import signature
from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import _C
from torch.types import Device
from . import _get_device_index, _get_nvml_device_index, _lazy_init, is_initialized
from ._memory_viz import memory as _memory, segments as _segments
from ._utils import _dummy_type
def _record_memory_history_impl(enabled: Optional[str]='all', context: Optional[str]='all', stacks: str='all', max_entries: int=sys.maxsize, device: Union[Device, int]=None):
    _C._cuda_record_memory_history(enabled, context, stacks, max_entries)