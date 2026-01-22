from math import log, sqrt
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.special import gammaln
from ..base import _fit_context
from ..utils import check_random_state
from ..utils._arpack import _init_arpack_v0
from ..utils._array_api import _convert_to_numpy, get_namespace
from ..utils._param_validation import Interval, RealNotInt, StrOptions
from ..utils.extmath import fast_logdet, randomized_svd, stable_cumsum, svd_flip
from ..utils.sparsefuncs import _implicit_column_offset, mean_variance_axis
from ..utils.validation import check_is_fitted
from ._base import _BasePCA
def _fit_full(self, X, n_components):
    """Fit the model by computing full SVD on X."""
    xp, is_array_api_compliant = get_namespace(X)
    n_samples, n_features = X.shape
    if n_components == 'mle':
        if n_samples < n_features:
            raise ValueError("n_components='mle' is only supported if n_samples >= n_features")
    elif not 0 <= n_components <= min(n_samples, n_features):
        raise ValueError("n_components=%r must be between 0 and min(n_samples, n_features)=%r with svd_solver='full'" % (n_components, min(n_samples, n_features)))
    self.mean_ = xp.mean(X, axis=0)
    X -= self.mean_
    if not is_array_api_compliant:
        U, S, Vt = linalg.svd(X, full_matrices=False)
    else:
        U, S, Vt = xp.linalg.svd(X, full_matrices=False)
    U, Vt = svd_flip(U, Vt)
    components_ = Vt
    explained_variance_ = S ** 2 / (n_samples - 1)
    total_var = xp.sum(explained_variance_)
    explained_variance_ratio_ = explained_variance_ / total_var
    singular_values_ = xp.asarray(S, copy=True)
    if n_components == 'mle':
        n_components = _infer_dimension(explained_variance_, n_samples)
    elif 0 < n_components < 1.0:
        if is_array_api_compliant:
            explained_variance_ratio_np = _convert_to_numpy(explained_variance_ratio_, xp=xp)
        else:
            explained_variance_ratio_np = explained_variance_ratio_
        ratio_cumsum = stable_cumsum(explained_variance_ratio_np)
        n_components = np.searchsorted(ratio_cumsum, n_components, side='right') + 1
    if n_components < min(n_features, n_samples):
        self.noise_variance_ = xp.mean(explained_variance_[n_components:])
    else:
        self.noise_variance_ = 0.0
    self.n_samples_ = n_samples
    self.components_ = components_[:n_components, :]
    self.n_components_ = n_components
    self.explained_variance_ = explained_variance_[:n_components]
    self.explained_variance_ratio_ = explained_variance_ratio_[:n_components]
    self.singular_values_ = singular_values_[:n_components]
    return (U, S, Vt)