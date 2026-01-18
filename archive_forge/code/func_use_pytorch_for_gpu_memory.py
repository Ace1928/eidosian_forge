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
def use_pytorch_for_gpu_memory() -> None:
    """Route GPU memory allocation via PyTorch.

    This is recommended for using PyTorch and cupy together, as otherwise
    OOM errors can occur when there's available memory sitting in the other
    library's pool.

    We'd like to support routing Tensorflow memory allocation via PyTorch as well
    (or vice versa), but do not currently have an implementation for it.
    """
    assert_pytorch_installed()
    if get_torch_default_device().type != 'cuda':
        return
    pools = context_pools.get()
    if 'pytorch' not in pools:
        pools['pytorch'] = cupy.cuda.MemoryPool(allocator=cupy_pytorch_allocator)
    cupy.cuda.set_allocator(pools['pytorch'].malloc)