import functools
import numpy as np
from numpy.core import overrides
def _setdiff1d_dispatcher(ar1, ar2, assume_unique=None):
    return (ar1, ar2)