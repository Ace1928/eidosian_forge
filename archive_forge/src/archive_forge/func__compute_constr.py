import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def _compute_constr(self, c_ineq, c_eq, s):
    return np.hstack((c_eq, c_ineq + s))