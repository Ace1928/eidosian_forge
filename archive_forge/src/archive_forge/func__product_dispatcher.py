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
def _product_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
    warnings.warn('`product` is deprecated as of NumPy 1.25.0, and will be removed in NumPy 2.0. Please use `prod` instead.', DeprecationWarning, stacklevel=3)
    return (a, out)