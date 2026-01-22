import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class AbsoluteError(BaseLoss):
    """Absolute error with identity link, for regression.

    Domain:
    y_true and y_pred all real numbers

    Link:
    y_pred = raw_prediction

    For a given sample x_i, the absolute error is defined as::

        loss(x_i) = |y_true_i - raw_prediction_i|

    Note that the exact hessian = 0 almost everywhere (except at one point, therefore
    differentiable = False). Optimization routines like in HGBT, however, need a
    hessian > 0. Therefore, we assign 1.
    """
    differentiable = False
    need_update_leaves_values = True

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyAbsoluteError(), link=IdentityLink())
        self.approx_hessian = True
        self.constant_hessian = sample_weight is None

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the weighted median of the target, i.e. over the samples
        axis=0.
        """
        if sample_weight is None:
            return np.median(y_true, axis=0)
        else:
            return _weighted_percentile(y_true, sample_weight, 50)