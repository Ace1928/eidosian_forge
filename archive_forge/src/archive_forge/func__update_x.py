import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
def _update_x(self, x):
    if not np.array_equal(x, self.x):
        _x = atleast_nd(x, ndim=1, xp=self.xp)
        self.x = self.xp.astype(_x, self.x_dtype)
        self.f_updated = False