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
@contextlib.contextmanager
def use_ops(name: str, **kwargs):
    """Change the backend to execute on for the scope of the block."""
    current_ops = get_current_ops()
    set_current_ops(get_ops(name, **kwargs))
    try:
        yield
    finally:
        set_current_ops(current_ops)