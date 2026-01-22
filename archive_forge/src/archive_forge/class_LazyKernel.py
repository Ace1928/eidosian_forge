import operator
import re
from collections import defaultdict
from functools import reduce
from pathlib import Path
from typing import Callable, Optional, Tuple
import numpy
from ..compat import cupy, has_cupy_gpu
class LazyKernel:
    """Wraps around `cupy.RawModule` and `cupy.RawKernel` to verify CuPy availability
    and lazily compile the latter on first invocation.

    The default CuPy behaviour triggers the compilation as soon as the `cupy.RawKernel` object
    is accessed."""
    name: str
    _kernel: Optional['cupy.RawKernel']
    _compile_callback: Optional[Callable[[], 'cupy.RawKernel']]
    __slots__ = ['name', '_kernel', '_compile_callback']

    def __init__(self, name: str, *, compile_callback: Optional[Callable[[], 'cupy.RawKernel']]=None) -> None:
        self.name = name
        self._kernel = None
        self._compile_callback = compile_callback

    def __call__(self, *args, **kwargs):
        self._compile_kernel()
        self._kernel(*args, **kwargs)

    def _compile_kernel(self):
        if self._kernel is not None:
            return
        if self._compile_callback is not None:
            self._kernel = self._compile_callback()
        elif KERNELS is not None:
            self._kernel = KERNELS.get_function(self.name)
        if self._kernel is None:
            raise ValueError(f"couldn't compile Cupy kernel '{self.name}'")