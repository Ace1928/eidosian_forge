from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
@primitive
def array_from_args(array_args, array_kwargs, *args):
    return _np.array(args, *array_args, **array_kwargs)