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
def _transpose_ex(a, axeses):
    """Transpose and diagonal

    Args:
        a
        axeses (sequence of sequences of ints)

    Returns:
        ndarray: a with its axes permutated. A writeable view is returned
        whenever possible.
    """
    shape = []
    strides = []
    for axes in axeses:
        shape.append(a.shape[axes[0]] if axes else 1)
        stride = sum((a.strides[axis] for axis in axes))
        strides.append(stride)
    a = a.view()
    a._set_shape_and_strides(shape, strides, True, True)
    return a