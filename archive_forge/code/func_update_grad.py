import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace
def update_grad():
    self._update_fun()
    self.ngev += 1
    self.g = approx_derivative(fun_wrapped, self.x, f0=self.f, **finite_diff_options)