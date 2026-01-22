import itertools
import time
import warnings
from abc import ABC
from math import sqrt
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .._config import config_context
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_random_state, gen_batches, metadata_routing
from ..utils._param_validation import (
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import (
from ._cdnmf_fast import _update_cdnmf_fast
def _minibatch_step(self, X, W, H, update_H):
    """Perform the update of W and H for one minibatch."""
    batch_size = X.shape[0]
    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = self._compute_regularization(X)
    if self.fresh_restarts or W is None:
        W = self._solve_W(X, H, self.fresh_restarts_max_iter)
    else:
        W, *_ = _multiplicative_update_w(X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma)
    if self._beta_loss < 1:
        W[W < np.finfo(np.float64).eps] = 0.0
    batch_cost = (_beta_divergence(X, W, H, self._beta_loss) + l1_reg_W * W.sum() + l1_reg_H * H.sum() + l2_reg_W * (W ** 2).sum() + l2_reg_H * (H ** 2).sum()) / batch_size
    if update_H:
        H[:] = _multiplicative_update_h(X, W, H, beta_loss=self._beta_loss, l1_reg_H=l1_reg_H, l2_reg_H=l2_reg_H, gamma=self._gamma, A=self._components_numerator, B=self._components_denominator, rho=self._rho)
        if self._beta_loss <= 1:
            H[H < np.finfo(np.float64).eps] = 0.0
    return batch_cost