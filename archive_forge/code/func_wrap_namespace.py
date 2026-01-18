from __future__ import absolute_import
import types
import warnings
from autograd.extend import primitive, notrace_primitive
import numpy as _np
import autograd.builtins as builtins
from numpy.core.einsumfunc import _parse_einsum_input
def wrap_namespace(old, new):
    unchanged_types = {float, int, type(None), type}
    int_types = {_np.int8, _np.int16, _np.int32, _np.int64, _np.integer}
    for name, obj in old.items():
        if obj in notrace_functions:
            new[name] = notrace_primitive(obj)
        elif callable(obj) and type(obj) is not type:
            new[name] = primitive(obj)
        elif type(obj) is type and obj in int_types:
            new[name] = wrap_intdtype(obj)
        elif type(obj) in unchanged_types:
            new[name] = obj