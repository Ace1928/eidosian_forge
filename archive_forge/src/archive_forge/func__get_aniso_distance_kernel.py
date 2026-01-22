import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
@cupy.memoize(for_each_device=True)
def _get_aniso_distance_kernel(int_type):
    """Returns kernel computing the Euclidean distance from coordinates with
    axis spacing != 1."""
    operation = _get_aniso_distance_kernel_code(int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(in_params='I z, I y, I x, raw F sampling', out_params='raw F dist', operation=operation, options=('--std=c++11',))