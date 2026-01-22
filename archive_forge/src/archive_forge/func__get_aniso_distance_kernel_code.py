import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import (_check_distances, _check_indices,
def _get_aniso_distance_kernel_code(int_type, raw_out_var=True):
    code = _generate_shape(ndim=3, int_type=int_type, var_name='dist', raw_var=raw_out_var)
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    code += _generate_aniso_distance_computation()
    return code