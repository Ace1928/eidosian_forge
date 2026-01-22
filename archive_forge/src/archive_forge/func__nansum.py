from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def _nansum(data, axis=None):
    if sparse.issparse(data):
        return np.sum(fillna(data, 0), axis=axis)
    else:
        return np.nansum(data, axis=axis)