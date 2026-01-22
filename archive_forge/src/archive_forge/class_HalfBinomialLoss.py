import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HalfBinomialLoss(BaseLoss):
    """Half Binomial deviance loss with logit link, for binary classification.

    This is also know as binary cross entropy, log-loss and logistic loss.

    Domain:
    y_true in [0, 1], i.e. regression on the unit interval
    y_pred in (0, 1), i.e. boundaries excluded

    Link:
    y_pred = expit(raw_prediction)

    For a given sample x_i, half Binomial deviance is defined as the negative
    log-likelihood of the Binomial/Bernoulli distribution and can be expressed
    as::

        loss(x_i) = log(1 + exp(raw_pred_i)) - y_true_i * raw_pred_i

    See The Elements of Statistical Learning, by Hastie, Tibshirani, Friedman,
    section 4.4.1 (about logistic regression).

    Note that the formulation works for classification, y = {0, 1}, as well as
    logistic regression, y = [0, 1].
    If you add `constant_to_optimal_zero` to the loss, you get half the
    Bernoulli/binomial deviance.

    More details: Inserting the predicted probability y_pred = expit(raw_prediction)
    in the loss gives the well known::

        loss(x_i) = - y_true_i * log(y_pred_i) - (1 - y_true_i) * log(1 - y_pred_i)
    """

    def __init__(self, sample_weight=None):
        super().__init__(closs=CyHalfBinomialLoss(), link=LogitLink(), n_classes=2)
        self.interval_y_true = Interval(0, 1, True, True)

    def constant_to_optimal_zero(self, y_true, sample_weight=None):
        term = xlogy(y_true, y_true) + xlogy(1 - y_true, 1 - y_true)
        if sample_weight is not None:
            term *= sample_weight
        return term

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples,) or (n_samples, 1)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, 2)
            Element-wise class probabilities.
        """
        if raw_prediction.ndim == 2 and raw_prediction.shape[1] == 1:
            raw_prediction = raw_prediction.squeeze(1)
        proba = np.empty((raw_prediction.shape[0], 2), dtype=raw_prediction.dtype)
        proba[:, 1] = self.link.inverse(raw_prediction)
        proba[:, 0] = 1 - proba[:, 1]
        return proba