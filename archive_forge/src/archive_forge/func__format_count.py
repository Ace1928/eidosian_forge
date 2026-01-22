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
def _format_count(cnt, pref_cnt):
    prefixes = [' ', 'K', 'M']
    prefix = prefixes[0]
    for new_prefix in prefixes[1:]:
        if pref_cnt < 750 * 1000:
            break
        prefix = new_prefix
        cnt //= 1000
        pref_cnt /= 1000
    return f'{cnt:7d} {prefix} '