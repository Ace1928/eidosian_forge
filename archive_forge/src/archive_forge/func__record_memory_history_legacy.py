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
def _record_memory_history_legacy(enabled: bool, record_context=True, trace_alloc_max_entries=1, trace_alloc_record_context=False, device: Union[Device, int]=None, record_context_cpp=False):
    _C._cuda_record_memory_history_legacy(enabled, record_context, trace_alloc_max_entries, trace_alloc_record_context, record_context_cpp)