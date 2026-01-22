import warnings
from numbers import Integral, Real
from operator import itemgetter
import numpy as np
import scipy.optimize
from scipy.linalg import cho_solve, cholesky, solve_triangular
from ..base import BaseEstimator, MultiOutputMixin, RegressorMixin, _fit_context, clone
from ..preprocessing._data import _handle_zeros_in_scale
from ..utils import check_random_state
from ..utils._param_validation import Interval, StrOptions
from ..utils.optimize import _check_optimize_result
from .kernels import RBF, Kernel
from .kernels import ConstantKernel as C
def _constrained_optimization(self, obj_func, initial_theta, bounds):
    if self.optimizer == 'fmin_l_bfgs_b':
        opt_res = scipy.optimize.minimize(obj_func, initial_theta, method='L-BFGS-B', jac=True, bounds=bounds)
        _check_optimize_result('lbfgs', opt_res)
        theta_opt, func_min = (opt_res.x, opt_res.fun)
    elif callable(self.optimizer):
        theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
    else:
        raise ValueError(f'Unknown optimizer {self.optimizer}.')
    return (theta_opt, func_min)