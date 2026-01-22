import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _sparse_linalg_cg(A, b, **kwargs):
    if 'rtol' in kwargs:
        kwargs['tol'] = kwargs.pop('rtol')
    if 'atol' not in kwargs:
        kwargs['atol'] = 'legacy'
    return scipy.sparse.linalg.cg(A, b, **kwargs)