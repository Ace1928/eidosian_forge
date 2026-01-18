import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import linalg
from scipy.linalg import pinvh
from ..base import RegressorMixin, _fit_context
from ..utils import _safe_indexing
from ..utils._param_validation import Hidden, Interval, StrOptions
from ..utils.extmath import fast_logdet
from ..utils.validation import _check_sample_weight
from ._base import LinearModel, _preprocess_data, _rescale_data
def update_coeff(X, y, coef_, alpha_, keep_lambda, sigma_):
    coef_[keep_lambda] = alpha_ * np.linalg.multi_dot([sigma_, X[:, keep_lambda].T, y])
    return coef_