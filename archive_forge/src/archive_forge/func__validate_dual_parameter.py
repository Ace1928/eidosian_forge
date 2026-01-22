import warnings
from numbers import Integral, Real
import numpy as np
from ..base import BaseEstimator, OutlierMixin, RegressorMixin, _fit_context
from ..linear_model._base import LinearClassifierMixin, LinearModel, SparseCoefMixin
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import _num_samples
from ._base import BaseLibSVM, BaseSVC, _fit_liblinear, _get_liblinear_solver_type
def _validate_dual_parameter(dual, loss, penalty, multi_class, X):
    """Helper function to assign the value of dual parameter."""
    if dual == 'auto':
        if X.shape[0] < X.shape[1]:
            try:
                _get_liblinear_solver_type(multi_class, penalty, loss, True)
                return True
            except ValueError:
                return False
        else:
            try:
                _get_liblinear_solver_type(multi_class, penalty, loss, False)
                return False
            except ValueError:
                return True
    elif dual == 'warn':
        warnings.warn("The default value of `dual` will change from `True` to `'auto'` in 1.5. Set the value of `dual` explicitly to suppress the warning.", FutureWarning)
        return True
    else:
        return dual