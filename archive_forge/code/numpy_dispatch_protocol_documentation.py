import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
Register NumPy array ufunc protocol.

    References
    ----------
    https://numpy.org/neps/nep-0013-ufunc-overrides.html
    