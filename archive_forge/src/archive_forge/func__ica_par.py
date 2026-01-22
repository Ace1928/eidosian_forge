import warnings
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import as_float_array, check_array, check_random_state
from ..utils._param_validation import Interval, Options, StrOptions, validate_params
from ..utils.validation import check_is_fitted
def _ica_par(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in range(max_iter):
        gwtx, g_wtx = g(np.dot(W, X), fun_args)
        W1 = _sym_decorrelation(np.dot(gwtx, X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        lim = max(abs(abs(np.einsum('ij,ij->i', W1, W)) - 1))
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn('FastICA did not converge. Consider increasing tolerance or the maximum number of iterations.', ConvergenceWarning)
    return (W, ii + 1)