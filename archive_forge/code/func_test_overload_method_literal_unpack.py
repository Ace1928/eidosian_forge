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
def test_overload_method_literal_unpack(self):

    @overload_method(types.Array, 'litfoo')
    def litfoo(arr, val):
        if isinstance(val, types.Integer):
            if not isinstance(val, types.Literal):

                def impl(arr, val):
                    return val
                return impl

    @njit
    def bar(A):
        return A.litfoo(51966)
    A = np.zeros(1)
    bar(A)
    self.assertEqual(bar(A), 51966)