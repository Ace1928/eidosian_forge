import builtins
import unittest
from numbers import Number
from functools import wraps
import numpy as np
from llvmlite import ir
import numba
from numba import njit, typeof, objmode
from numba.core import cgutils, types, typing
from numba.core.pythonapi import box
from numba.core.errors import TypingError
from numba.core.registry import cpu_target
from numba.extending import (intrinsic, lower_builtin, overload_classmethod,
from numba.np import numpy_support
from numba.tests.support import TestCase, MemoryLeakMixin
class MyArrayType(types.Array):

    def __init__(self, dtype, ndim, layout, readonly=False, aligned=True):
        name = f'MyArray({ndim}, {dtype}, {layout})'
        super().__init__(dtype, ndim, layout, readonly=readonly, aligned=aligned, name=name)

    def copy(self, *args, **kwargs):
        raise NotImplementedError

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            for inp in inputs:
                if not isinstance(inp, (types.Array, types.Number)):
                    return NotImplemented
            if all((isinstance(inp, MyArrayType) for inp in inputs)):
                return NotImplemented
            return MyArrayType
        else:
            return NotImplemented

    @property
    def box_type(self):
        return MyArray