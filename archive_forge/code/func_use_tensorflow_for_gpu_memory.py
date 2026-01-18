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
def use_tensorflow_for_gpu_memory() -> None:
    """Route GPU memory allocation via TensorFlow.

    This is recommended for using TensorFlow and cupy together, as otherwise
    OOM errors can occur when there's available memory sitting in the other
    library's pool.

    We'd like to support routing PyTorch memory allocation via Tensorflow as
    well (or vice versa), but do not currently have an implementation for it.
    """
    assert_tensorflow_installed()
    pools = context_pools.get()
    if 'tensorflow' not in pools:
        pools['tensorflow'] = cupy.cuda.MemoryPool(allocator=cupy_tensorflow_allocator)
    cupy.cuda.set_allocator(pools['tensorflow'].malloc)