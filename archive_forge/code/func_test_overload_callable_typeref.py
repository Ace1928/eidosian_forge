import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
def test_overload_callable_typeref(self):

    @overload(CallableTypeRef)
    def callable_type_call_ovld1(x):
        if isinstance(x, types.Integer):

            def impl(x):
                return 42.5 + x
            return impl

    @overload(CallableTypeRef)
    def callable_type_call_ovld2(x):
        if isinstance(x, types.UnicodeType):

            def impl(x):
                return '42.5' + x
            return impl

    @njit
    def foo(a, b):
        return (MyClass(a), MyClass(b))
    args = (4, '4')
    expected = (42.5 + args[0], '42.5' + args[1])
    self.assertPreciseEqual(foo(*args), expected)