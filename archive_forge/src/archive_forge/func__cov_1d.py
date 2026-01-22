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
@conditional_jit(cache=True, nopython=True)
def _cov_1d(x):
    x = x - x.mean()
    ddof = x.shape[0] - 1
    return np.dot(x.T, x.conj()) / ddof