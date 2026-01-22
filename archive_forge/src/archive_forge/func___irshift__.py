import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
def __irshift__(self, other):
    return self._inplace_op(cupy.right_shift, other)