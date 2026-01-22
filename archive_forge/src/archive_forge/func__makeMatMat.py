import warnings
import numpy as np
from scipy.linalg import (inv, eigh, cho_factor, cho_solve,
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import issparse
def _makeMatMat(m):
    if m is None:
        return None
    elif callable(m):
        return lambda v: m(v)
    else:
        return lambda v: m @ v