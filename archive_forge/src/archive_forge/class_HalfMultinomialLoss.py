import numbers
import numpy as np
from scipy.special import xlogy
from ..utils import check_scalar
from ..utils.stats import _weighted_percentile
from ._loss import (
from .link import (
class HalfMultinomialLoss(BaseLoss):
    """Categorical cross-entropy loss, for multiclass classification.

    Domain:
    y_true in {0, 1, 2, 3, .., n_classes - 1}
    y_pred has n_classes elements, each element in (0, 1)

    Link:
    y_pred = softmax(raw_prediction)

    Note: We assume y_true to be already label encoded. The inverse link is
    softmax. But the full link function is the symmetric multinomial logit
    function.

    For a given sample x_i, the categorical cross-entropy loss is defined as
    the negative log-likelihood of the multinomial distribution, it
    generalizes the binary cross-entropy to more than 2 classes::

        loss_i = log(sum(exp(raw_pred_{i, k}), k=0..n_classes-1))
                - sum(y_true_{i, k} * raw_pred_{i, k}, k=0..n_classes-1)

    See [1].

    Note that for the hessian, we calculate only the diagonal part in the
    classes: If the full hessian for classes k and l and sample i is H_i_k_l,
    we calculate H_i_k_k, i.e. k=l.

    Reference
    ---------
    .. [1] :arxiv:`Simon, Noah, J. Friedman and T. Hastie.
        "A Blockwise Descent Algorithm for Group-penalized Multiresponse and
        Multinomial Regression".
        <1311.6529>`
    """
    is_multiclass = True

    def __init__(self, sample_weight=None, n_classes=3):
        super().__init__(closs=CyHalfMultinomialLoss(), link=MultinomialLogit(), n_classes=n_classes)
        self.interval_y_true = Interval(0, np.inf, True, False)
        self.interval_y_pred = Interval(0, 1, False, False)

    def in_y_true_range(self, y):
        """Return True if y is in the valid range of y_true.

        Parameters
        ----------
        y : ndarray
        """
        return self.interval_y_true.includes(y) and np.all(y.astype(int) == y)

    def fit_intercept_only(self, y_true, sample_weight=None):
        """Compute raw_prediction of an intercept-only model.

        This is the softmax of the weighted average of the target, i.e. over
        the samples axis=0.
        """
        out = np.zeros(self.n_classes, dtype=y_true.dtype)
        eps = np.finfo(y_true.dtype).eps
        for k in range(self.n_classes):
            out[k] = np.average(y_true == k, weights=sample_weight, axis=0)
            out[k] = np.clip(out[k], eps, 1 - eps)
        return self.link.link(out[None, :]).reshape(-1)

    def predict_proba(self, raw_prediction):
        """Predict probabilities.

        Parameters
        ----------
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        return self.link.inverse(raw_prediction)

    def gradient_proba(self, y_true, raw_prediction, sample_weight=None, gradient_out=None, proba_out=None, n_threads=1):
        """Compute gradient and class probabilities fow raw_prediction.

        Parameters
        ----------
        y_true : C-contiguous array of shape (n_samples,)
            Observed, true target values.
        raw_prediction : array of shape (n_samples, n_classes)
            Raw prediction values (in link space).
        sample_weight : None or C-contiguous array of shape (n_samples,)
            Sample weights.
        gradient_out : None or array of shape (n_samples, n_classes)
            A location into which the gradient is stored. If None, a new array
            might be created.
        proba_out : None or array of shape (n_samples, n_classes)
            A location into which the class probabilities are stored. If None,
            a new array might be created.
        n_threads : int, default=1
            Might use openmp thread parallelism.

        Returns
        -------
        gradient : array of shape (n_samples, n_classes)
            Element-wise gradients.

        proba : array of shape (n_samples, n_classes)
            Element-wise class probabilities.
        """
        if gradient_out is None:
            if proba_out is None:
                gradient_out = np.empty_like(raw_prediction)
                proba_out = np.empty_like(raw_prediction)
            else:
                gradient_out = np.empty_like(proba_out)
        elif proba_out is None:
            proba_out = np.empty_like(gradient_out)
        self.closs.gradient_proba(y_true=y_true, raw_prediction=raw_prediction, sample_weight=sample_weight, gradient_out=gradient_out, proba_out=proba_out, n_threads=n_threads)
        return (gradient_out, proba_out)