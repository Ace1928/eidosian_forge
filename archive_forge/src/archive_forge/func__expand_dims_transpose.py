import copy
import itertools
import operator
import string
import warnings
import cupy
from cupy._core import _accelerator
from cupy import _util
from cupy.linalg._einsum_opt import _greedy_path
from cupy.linalg._einsum_opt import _optimal_path
from cupy.linalg._einsum_cutn import _try_use_cutensornet
def _expand_dims_transpose(arr, mode, mode_out):
    """Return a reshaped and transposed array.

    The input array ``arr`` having ``mode`` as its modes is reshaped and
    transposed so that modes of the output becomes ``mode_out``.

    Example
        >>> import cupy
        >>> a = cupy.zeros((10, 20))
        >>> mode_a = ('A', 'B')
        >>> mode_out = ('B', 'C', 'A')
        >>> out = cupy.linalg.einsum._expand_dims_transpose(a, mode_a,
        ...                                                 mode_out)
        >>> out.shape
        (20, 1, 10)

    Args:
        arr (cupy.ndarray):
        mode (tuple or list): The modes of input array.
        mode_out (tuple or list): The modes of output array.

    Returns:
        cupy.ndarray: The reshaped and transposed array.

    """
    mode = list(mode)
    shape = list(arr.shape)
    axes = []
    for i in mode_out:
        if i not in mode:
            mode.append(i)
            shape.append(1)
        axes.append(mode.index(i))
    return cupy.transpose(arr.reshape(shape), axes)