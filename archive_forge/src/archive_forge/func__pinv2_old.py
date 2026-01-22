import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
from scipy.linalg import svd
from ..base import (
from ..exceptions import ConvergenceWarning
from ..utils import check_array, check_consistent_length
from ..utils._param_validation import Interval, StrOptions
from ..utils.extmath import svd_flip
from ..utils.fixes import parse_version, sp_version
from ..utils.validation import FLOAT_DTYPES, check_is_fitted
def _pinv2_old(a):
    u, s, vh = svd(a, full_matrices=False, check_finite=False)
    t = u.dtype.char.lower()
    factor = {'f': 1000.0, 'd': 1000000.0}
    cond = np.max(s) * factor[t] * np.finfo(t).eps
    rank = np.sum(s > cond)
    u = u[:, :rank]
    u /= s[:rank]
    return np.transpose(np.conjugate(np.dot(u, vh[:rank])))