import functools
from typing import Callable, Final, Optional
import numpy
import cupy
class _OpMode:
    func: cupy._core._kernel.ufunc
    numpy_func: numpy.ufunc
    idempotent: bool
    identity_of: Callable

    def __init__(self, func_name: str, idempotent: bool, identity_of: Callable) -> None:
        try:
            self.func = getattr(cupy, func_name)
            self.numpy_func = getattr(numpy, func_name)
        except AttributeError:
            raise RuntimeError('No such function exists')
        self.idempotent = idempotent
        self.identity_of = identity_of