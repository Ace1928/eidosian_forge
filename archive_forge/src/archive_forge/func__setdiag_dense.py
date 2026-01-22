import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _setdiag_dense(m, d):
    step = len(d) + 1
    m.flat[::step] = d