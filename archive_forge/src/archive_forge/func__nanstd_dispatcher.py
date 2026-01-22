import functools
import warnings
import numpy as np
from numpy.lib import function_base
from numpy.core import overrides
def _nanstd_dispatcher(a, axis=None, dtype=None, out=None, ddof=None, keepdims=None, *, where=None):
    return (a, out)