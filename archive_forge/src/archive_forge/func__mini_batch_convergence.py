import warnings
from abc import ABC, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from ..base import (
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import _euclidean_distances, euclidean_distances
from ..utils import check_array, check_random_state
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.fixes import threadpool_info, threadpool_limits
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.validation import (
from ._k_means_common import (
from ._k_means_elkan import (
from ._k_means_lloyd import lloyd_iter_chunked_dense, lloyd_iter_chunked_sparse
from ._k_means_minibatch import _minibatch_update_dense, _minibatch_update_sparse
def _mini_batch_convergence(self, step, n_steps, n_samples, centers_squared_diff, batch_inertia):
    """Helper function to encapsulate the early stopping logic"""
    batch_inertia /= self._batch_size
    step = step + 1
    if step == 1:
        if self.verbose:
            print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}')
        return False
    if self._ewa_inertia is None:
        self._ewa_inertia = batch_inertia
    else:
        alpha = self._batch_size * 2.0 / (n_samples + 1)
        alpha = min(alpha, 1)
        self._ewa_inertia = self._ewa_inertia * (1 - alpha) + batch_inertia * alpha
    if self.verbose:
        print(f'Minibatch step {step}/{n_steps}: mean batch inertia: {batch_inertia}, ewa inertia: {self._ewa_inertia}')
    if self._tol > 0.0 and centers_squared_diff <= self._tol:
        if self.verbose:
            print(f'Converged (small centers change) at step {step}/{n_steps}')
        return True
    if self._ewa_inertia_min is None or self._ewa_inertia < self._ewa_inertia_min:
        self._no_improvement = 0
        self._ewa_inertia_min = self._ewa_inertia
    else:
        self._no_improvement += 1
    if self.max_no_improvement is not None and self._no_improvement >= self.max_no_improvement:
        if self.verbose:
            print(f'Converged (lack of improvement in inertia) at step {step}/{n_steps}')
        return True
    return False