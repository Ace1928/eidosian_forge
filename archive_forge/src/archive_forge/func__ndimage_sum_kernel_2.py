import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util
def _ndimage_sum_kernel_2(input, labels, index, sum_val, batch_size=4):
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape((-1,) + (1,) * input.ndim)
        sum_axes = tuple(range(1, 1 + input.ndim))
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(axis=sum_axes)
    return sum_val