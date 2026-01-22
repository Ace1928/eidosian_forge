import numbers
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from numbers import Integral, Real
import numpy as np
from scipy import linalg, optimize, sparse
from scipy.sparse import linalg as sp_linalg
from ..base import MultiOutputMixin, RegressorMixin, _fit_context, is_classifier
from ..exceptions import ConvergenceWarning
from ..metrics import check_scoring, get_scorer_names
from ..model_selection import GridSearchCV
from ..preprocessing import LabelBinarizer
from ..utils import (
from ..utils._param_validation import Interval, StrOptions, validate_params
from ..utils.extmath import row_norms, safe_sparse_dot
from ..utils.fixes import _sparse_linalg_cg
from ..utils.metadata_routing import (
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, LinearModel, _preprocess_data, _rescale_data
from ._sag import sag_solver
def _eigen_decompose_gram(self, X, y, sqrt_sw):
    """Eigendecomposition of X.X^T, used when n_samples <= n_features."""
    K, X_mean = self._compute_gram(X, sqrt_sw)
    if self.fit_intercept:
        K += np.outer(sqrt_sw, sqrt_sw)
    eigvals, Q = linalg.eigh(K)
    QT_y = np.dot(Q.T, y)
    return (X_mean, eigvals, Q, QT_y)