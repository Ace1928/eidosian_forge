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
def _fit_calibrator(clf, predictions, y, classes, method, sample_weight=None):
    """Fit calibrator(s) and return a `_CalibratedClassifier`
    instance.

    `n_classes` (i.e. `len(clf.classes_)`) calibrators are fitted.
    However, if `n_classes` equals 2, one calibrator is fitted.

    Parameters
    ----------
    clf : estimator instance
        Fitted classifier.

    predictions : array-like, shape (n_samples, n_classes) or (n_samples, 1)                     when binary.
        Raw predictions returned by the un-calibrated base classifier.

    y : array-like, shape (n_samples,)
        The targets.

    classes : ndarray, shape (n_classes,)
        All the prediction classes.

    method : {'sigmoid', 'isotonic'}
        The method to use for calibration.

    sample_weight : ndarray, shape (n_samples,), default=None
        Sample weights. If None, then samples are equally weighted.

    Returns
    -------
    pipeline : _CalibratedClassifier instance
    """
    Y = label_binarize(y, classes=classes)
    label_encoder = LabelEncoder().fit(classes)
    pos_class_indices = label_encoder.transform(clf.classes_)
    calibrators = []
    for class_idx, this_pred in zip(pos_class_indices, predictions.T):
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            calibrator = _SigmoidCalibration()
        calibrator.fit(this_pred, Y[:, class_idx], sample_weight)
        calibrators.append(calibrator)
    pipeline = _CalibratedClassifier(clf, calibrators, method=method, classes=classes)
    return pipeline