import warnings
from inspect import signature
from math import log
from numbers import Integral, Real
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.utils import Bunch
from ._loss import HalfBinomialLoss
from .base import (
from .isotonic import IsotonicRegression
from .model_selection import check_cv, cross_val_predict
from .preprocessing import LabelEncoder, label_binarize
from .svm import LinearSVC
from .utils import (
from .utils._param_validation import (
from .utils._plotting import _BinaryClassifierCurveDisplayMixin
from .utils._response import _get_response_values, _process_predict_proba
from .utils.metadata_routing import (
from .utils.multiclass import check_classification_targets
from .utils.parallel import Parallel, delayed
from .utils.validation import (
def _sigmoid_calibration(predictions, y, sample_weight=None, max_abs_prediction_threshold=30):
    """Probability Calibration with sigmoid method (Platt 2000)

    Parameters
    ----------
    predictions : ndarray of shape (n_samples,)
        The decision function or predict proba for the samples.

    y : ndarray of shape (n_samples,)
        The targets.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    a : float
        The slope.

    b : float
        The intercept.

    References
    ----------
    Platt, "Probabilistic Outputs for Support Vector Machines"
    """
    predictions = column_or_1d(predictions)
    y = column_or_1d(y)
    F = predictions
    scale_constant = 1.0
    max_prediction = np.max(np.abs(F))
    if max_prediction >= max_abs_prediction_threshold:
        scale_constant = max_prediction
        F = F / scale_constant
    mask_negative_samples = y <= 0
    if sample_weight is not None:
        prior0 = sample_weight[mask_negative_samples].sum()
        prior1 = sample_weight[~mask_negative_samples].sum()
    else:
        prior0 = float(np.sum(mask_negative_samples))
        prior1 = y.shape[0] - prior0
    T = np.zeros_like(y, dtype=predictions.dtype)
    T[y > 0] = (prior1 + 1.0) / (prior1 + 2.0)
    T[y <= 0] = 1.0 / (prior0 + 2.0)
    bin_loss = HalfBinomialLoss()

    def loss_grad(AB):
        raw_prediction = -(AB[0] * F + AB[1]).astype(dtype=predictions.dtype)
        l, g = bin_loss.loss_gradient(y_true=T, raw_prediction=raw_prediction, sample_weight=sample_weight)
        loss = l.sum()
        grad = np.asarray([-g @ F, -g.sum()], dtype=np.float64)
        return (loss, grad)
    AB0 = np.array([0.0, log((prior0 + 1.0) / (prior1 + 1.0))])
    opt_result = minimize(loss_grad, AB0, method='L-BFGS-B', jac=True, options={'gtol': 1e-06, 'ftol': 64 * np.finfo(float).eps})
    AB_ = opt_result.x
    return (AB_[0] / scale_constant, AB_[1])