import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HalfPoissonLoss(BaseLoss):
    """Half Poisson deviance loss with log-link, for regression.

    Domain:
    y_true in non-negative real numbers
    y_pred in positive real numbers

    Link:
    y_pred = exp(raw_prediction)

    For a given sample x_i, half the Poisson deviance is defined as::

        loss(x_i) = y_true_i * log(y_true_i/exp(raw_prediction_i))
                    - y_true_i + exp(raw_prediction_i)

    Half the Poisson deviance is actually the negative log-likelihood up to
    constant terms (not involving raw_prediction) and simplifies the
    computation of the gradients.
    We also skip the constant term `y_true_i * log(y_true_i) - y_true_i`.
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfPoissonLoss(), link=LogLink())
        self.interval_y_true = Interval(0, np.inf, True, False)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = xlogy(y_true, y_true) - y_true
        if sample_weight is not None:
            term *= sample_weight
        return term