from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
def wrap_if_boxes_inside(raw_array, slow_op_name=None):
    if raw_array.dtype is _np.dtype('O'):
        if slow_op_name:
            warnings.warn('{0} is slow for array inputs. np.concatenate() is faster.'.format(slow_op_name))
        return array_from_args((), {}, *raw_array.ravel()).reshape(raw_array.shape)
    else:
        return raw_array