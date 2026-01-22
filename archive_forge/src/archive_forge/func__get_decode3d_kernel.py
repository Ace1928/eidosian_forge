import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
@cupy.memoize(for_each_device=True)
def _get_decode3d_kernel(size_max):
    """Unpack 3 coordinates encoded as a single integer."""
    code = _get_decode3d_code(size_max, int_type='')
    return cupy.ElementwiseKernel(in_params='E encoded', out_params='I x, I y, I z', operation=code, options=('--std=c++11',))