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
def _solve_W(self, X, H, max_iter):
    """Minimize the objective function w.r.t W.

        Update W with H being fixed, until convergence. This is the heart
        of `transform` but it's also used during `fit` when doing fresh restarts.
        """
    avg = np.sqrt(X.mean() / self._n_components)
    W = np.full((X.shape[0], self._n_components), avg, dtype=X.dtype)
    W_buffer = W.copy()
    l1_reg_W, _, l2_reg_W, _ = self._compute_regularization(X)
    for _ in range(max_iter):
        W, *_ = _multiplicative_update_w(X, W, H, self._beta_loss, l1_reg_W, l2_reg_W, self._gamma)
        W_diff = linalg.norm(W - W_buffer) / linalg.norm(W)
        if self.tol > 0 and W_diff <= self.tol:
            break
        W_buffer[:] = W
    return W