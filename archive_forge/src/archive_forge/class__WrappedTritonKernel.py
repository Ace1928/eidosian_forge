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
class _WrappedTritonKernel:
    """Just a simple wrapper to store some metadata for testing purposes."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.kernel_invoked = False

    def __call__(self, *args, **kwargs):
        res = self.kernel(*args, **kwargs)
        self.kernel_invoked = True
        return res