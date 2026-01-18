import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def two_de(x):
    """Jitting numpy at_least_2d."""
    if not isinstance(x, np.ndarray):
        return np.atleast_2d(x)
    if x.ndim == 0:
        result = x.reshape(1, 1)
    elif x.ndim == 1:
        result = x[newaxis, :]
    else:
        result = x
    return result