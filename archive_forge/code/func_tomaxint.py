import atexit
import binascii
import functools
import hashlib
import operator
import os
import time
import numpy
import warnings
from numpy.linalg import LinAlgError
import cupy
from cupy import _core
from cupy import cuda
from cupy.cuda import curand
from cupy.cuda import device
from cupy.random import _kernels
from cupy import _util
import cupyx
def tomaxint(self, size=None):
    """Draws integers between 0 and max integer inclusive.

        Return a sample of uniformly distributed random integers in the
        interval [0, ``np.iinfo(np.int_).max``]. The `np.int_` type translates
        to the C long integer type and its precision is platform dependent.

        Args:
            size (int or tuple of ints): Output shape.

        Returns:
            cupy.ndarray: Drawn samples.

        .. seealso::
            :meth:`numpy.random.RandomState.tomaxint`

        """
    if size is None:
        size = ()
    sample = cupy.empty(size, dtype=cupy.int_)
    size_in_int = sample.dtype.itemsize // 4
    curand.generate(self._generator, sample.data.ptr, sample.size * size_in_int)
    sample &= cupy.iinfo(cupy.int_).max
    return sample