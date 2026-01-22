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
def _parse_visible_devices() -> Union[List[int], List[str]]:
    """Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv('CUDA_VISIBLE_DEVICES')
    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in '+-')):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> List[str]:
        rcs: List[str] = []
        for elem in lst.split(','):
            if elem in rcs:
                return cast(List[str], [])
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs
    if var.startswith('GPU-'):
        return parse_list_with_prefix(var, 'GPU-')
    if var.startswith('MIG-'):
        return parse_list_with_prefix(var, 'MIG-')
    rc: List[int] = []
    for elem in var.split(','):
        x = _strtoul(elem.strip())
        if x in rc:
            return cast(List[int], [])
        if x < 0:
            break
        rc.append(x)
    return rc