import warnings
from abc import ABCMeta, abstractmethod
from itertools import chain
from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics import accuracy_score, r2_score
from ..model_selection import train_test_split
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, Options, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import (
from ..utils.optimize import _check_optimize_result
from ..utils.validation import check_is_fitted
from ._base import ACTIVATIONS, DERIVATIVES, LOSS_FUNCTIONS
from ._stochastic_optimizers import AdamOptimizer, SGDOptimizer
def _fit_lbfgs(self, X, y, activations, deltas, coef_grads, intercept_grads, layer_units):
    self._coef_indptr = []
    self._intercept_indptr = []
    start = 0
    for i in range(self.n_layers_ - 1):
        n_fan_in, n_fan_out = (layer_units[i], layer_units[i + 1])
        end = start + n_fan_in * n_fan_out
        self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
        start = end
    for i in range(self.n_layers_ - 1):
        end = start + layer_units[i + 1]
        self._intercept_indptr.append((start, end))
        start = end
    packed_coef_inter = _pack(self.coefs_, self.intercepts_)
    if self.verbose is True or self.verbose >= 1:
        iprint = 1
    else:
        iprint = -1
    opt_res = scipy.optimize.minimize(self._loss_grad_lbfgs, packed_coef_inter, method='L-BFGS-B', jac=True, options={'maxfun': self.max_fun, 'maxiter': self.max_iter, 'iprint': iprint, 'gtol': self.tol}, args=(X, y, activations, deltas, coef_grads, intercept_grads))
    self.n_iter_ = _check_optimize_result('lbfgs', opt_res, self.max_iter)
    self.loss_ = opt_res.fun
    self._unpack(opt_res.x)