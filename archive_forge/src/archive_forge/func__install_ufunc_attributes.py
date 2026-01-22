import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def _install_ufunc_attributes(self, template) -> None:

    def get_attr_fn(attr: str) -> Callable:

        def impl(ufunc):
            val = getattr(ufunc.key[0], attr)
            return lambda ufunc: val
        return impl
    at = types.Function(template)
    attributes = ('nin', 'nout', 'nargs', 'identity', 'signature')
    for attr in attributes:
        attr_fn = get_attr_fn(attr)
        overload_attribute(at, attr)(attr_fn)