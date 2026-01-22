import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def _sequence_of_arrays(context, func_name, arrays, dim_chooser=_homogeneous_dims):
    if not isinstance(arrays, types.BaseTuple) or not len(arrays) or (not all((isinstance(a, types.Array) for a in arrays))):
        raise TypeError('%s(): expecting a non-empty tuple of arrays, got %s' % (func_name, arrays))
    ndim = dim_chooser(context, func_name, arrays)
    dtype = context.unify_types(*(a.dtype for a in arrays))
    if dtype is None:
        raise TypeError('%s(): input arrays must have compatible dtypes' % func_name)
    return (dtype, ndim)