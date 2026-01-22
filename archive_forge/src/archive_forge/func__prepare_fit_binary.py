import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from ..base import (
from ..exceptions import ConvergenceWarning
from ..model_selection import ShuffleSplit, StratifiedShuffleSplit
from ..utils import check_random_state, compute_class_weight, deprecated
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import safe_sparse_dot
from ..utils.metaestimators import available_if
from ..utils.multiclass import _check_partial_fit_first_call
from ..utils.parallel import Parallel, delayed
from ..utils.validation import _check_sample_weight, check_is_fitted
from ._base import LinearClassifierMixin, SparseCoefMixin, make_dataset
from ._sgd_fast import (
def _prepare_fit_binary(est, y, i, input_dtye):
    """Initialization for fit_binary.

    Returns y, coef, intercept, average_coef, average_intercept.
    """
    y_i = np.ones(y.shape, dtype=input_dtye, order='C')
    y_i[y != est.classes_[i]] = -1.0
    average_intercept = 0
    average_coef = None
    if len(est.classes_) == 2:
        if not est.average:
            coef = est.coef_.ravel()
            intercept = est.intercept_[0]
        else:
            coef = est._standard_coef.ravel()
            intercept = est._standard_intercept[0]
            average_coef = est._average_coef.ravel()
            average_intercept = est._average_intercept[0]
    elif not est.average:
        coef = est.coef_[i]
        intercept = est.intercept_[i]
    else:
        coef = est._standard_coef[i]
        intercept = est._standard_intercept[i]
        average_coef = est._average_coef[i]
        average_intercept = est._average_intercept[i]
    return (y_i, coef, intercept, average_coef, average_intercept)