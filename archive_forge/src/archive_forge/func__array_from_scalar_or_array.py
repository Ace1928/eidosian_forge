from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
@primitive
def _array_from_scalar_or_array(array_args, array_kwargs, scalar):
    return _np.array(scalar, *array_args, **array_kwargs)