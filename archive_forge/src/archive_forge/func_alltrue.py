import functools
import types
import warnings
import numpy as np
from .._utils import set_module
from . import multiarray as mu
from . import overrides
from . import umath as um
from . import numerictypes as nt
from .multiarray import asarray, array, asanyarray, concatenate
from . import _methods
@array_function_dispatch(_alltrue_dispatcher, verify=False)
def alltrue(*args, **kwargs):
    """
    Check if all elements of input array are true.

    .. deprecated:: 1.25.0
        ``alltrue`` is deprecated as of NumPy 1.25.0, and will be
        removed in NumPy 2.0. Please use `all` instead.

    See Also
    --------
    numpy.all : Equivalent function; see for details.
    """
    return all(*args, **kwargs)