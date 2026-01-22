import collections.abc
import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupy._core._gufuncs import _GUFunc
from cupy.linalg import _solve
from cupy.linalg import _util
Returns the kronecker product of two arrays.

    Args:
        a (~cupy.ndarray): The first argument.
        b (~cupy.ndarray): The second argument.

    Returns:
        ~cupy.ndarray: Output array.

    .. seealso:: :func:`numpy.kron`

    