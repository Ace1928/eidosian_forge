from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def _check_numpy_dtype(x):
    try:
        if all([len(xi) == len(x[0]) for xi in x]):
            return None
        else:
            return object
    except TypeError as e:
        if str(e).startswith('sparse matrix length is ambiguous'):
            return object
        elif str(e).endswith('has no len()'):
            if any([hasattr(xi, '__len__') for xi in x]):
                return object
            else:
                return None
        else:
            raise