from numbers import Integral, Real
import numpy as np
import scipy.optimize
from ..._loss.loss import (
from ...base import BaseEstimator, RegressorMixin, _fit_context
from ...utils import check_array
from ...utils._openmp_helpers import _openmp_effective_n_threads
from ...utils._param_validation import Hidden, Interval, StrOptions
from ...utils.optimize import _check_optimize_result
from ...utils.validation import _check_sample_weight, check_is_fitted
from .._linear_loss import LinearModelLoss
from ._newton_solver import NewtonCholeskySolver, NewtonSolver
def _linear_predictor(self, X):
    """Compute the linear_predictor = `X @ coef_ + intercept_`.

        Note that we often use the term raw_prediction instead of linear predictor.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values of linear predictor.
        """
    check_is_fitted(self)
    X = self._validate_data(X, accept_sparse=['csr', 'csc', 'coo'], dtype=[np.float64, np.float32], ensure_2d=True, allow_nd=False, reset=False)
    return X @ self.coef_ + self.intercept_