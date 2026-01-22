import contextlib
import threading
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, Type, cast
from .. import registry
from ..compat import cupy, has_cupy
from ..util import (
from ._cupy_allocators import cupy_pytorch_allocator, cupy_tensorflow_allocator
from ._param_server import ParamServer
from .cupy_ops import CupyOps
from .mps_ops import MPSOps
from .numpy_ops import NumpyOps
from .ops import Ops
def _import_extra_cpu_backends():
    try:
        from thinc_apple_ops import AppleOps
    except ImportError:
        pass
    try:
        from thinc_bigendian_ops import BigEndianOps
    except ImportError:
        pass