import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
def _partition_factory(pivotimpl, argpartition=False):

    def _partition(A, low, high, I=None):
        mid = low + high >> 1
        if pivotimpl(A[mid], A[low]):
            A[low], A[mid] = (A[mid], A[low])
            if argpartition:
                I[low], I[mid] = (I[mid], I[low])
        if pivotimpl(A[high], A[mid]):
            A[high], A[mid] = (A[mid], A[high])
            if argpartition:
                I[high], I[mid] = (I[mid], I[high])
        if pivotimpl(A[mid], A[low]):
            A[low], A[mid] = (A[mid], A[low])
            if argpartition:
                I[low], I[mid] = (I[mid], I[low])
        pivot = A[mid]
        A[high], A[mid] = (A[mid], A[high])
        if argpartition:
            I[high], I[mid] = (I[mid], I[high])
        i = low
        j = high - 1
        while True:
            while i < high and pivotimpl(A[i], pivot):
                i += 1
            while j >= low and pivotimpl(pivot, A[j]):
                j -= 1
            if i >= j:
                break
            A[i], A[j] = (A[j], A[i])
            if argpartition:
                I[i], I[j] = (I[j], I[i])
            i += 1
            j -= 1
        A[i], A[high] = (A[high], A[i])
        if argpartition:
            I[i], I[high] = (I[high], I[i])
        return i
    return _partition