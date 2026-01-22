from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from ..base import (
from ..exceptions import NotFittedError
from ..metrics.pairwise import pairwise_kernels
from ..preprocessing import KernelCenterer
from ..utils._arpack import _init_arpack_v0
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import _randomized_eigsh, svd_flip
from ..utils.validation import (
def _fit_inverse_transform(self, X_transformed, X):
    if hasattr(X, 'tocsr'):
        raise NotImplementedError('Inverse transform not implemented for sparse matrices!')
    n_samples = X_transformed.shape[0]
    K = self._get_kernel(X_transformed)
    K.flat[::n_samples + 1] += self.alpha
    self.dual_coef_ = linalg.solve(K, X, assume_a='pos', overwrite_a=True)
    self.X_transformed_fit_ = X_transformed