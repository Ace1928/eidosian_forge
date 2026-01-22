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
class _CalibratedClassifier:
    """Pipeline-like chaining a fitted classifier and its fitted calibrators.

    Parameters
    ----------
    estimator : estimator instance
        Fitted classifier.

    calibrators : list of fitted estimator instances
        List of fitted calibrators (either 'IsotonicRegression' or
        '_SigmoidCalibration'). The number of calibrators equals the number of
        classes. However, if there are 2 classes, the list contains only one
        fitted calibrator.

    classes : array-like of shape (n_classes,)
        All the prediction classes.

    method : {'sigmoid', 'isotonic'}, default='sigmoid'
        The method to use for calibration. Can be 'sigmoid' which
        corresponds to Platt's method or 'isotonic' which is a
        non-parametric approach based on isotonic regression.
    """

    def __init__(self, estimator, calibrators, *, classes, method='sigmoid'):
        self.estimator = estimator
        self.calibrators = calibrators
        self.classes = classes
        self.method = method

    def predict_proba(self, X):
        """Calculate calibrated probabilities.

        Calculates classification calibrated probabilities
        for each class, in a one-vs-all manner, for `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The sample data.

        Returns
        -------
        proba : array, shape (n_samples, n_classes)
            The predicted probabilities. Can be exact zeros.
        """
        predictions, _ = _get_response_values(self.estimator, X, response_method=['decision_function', 'predict_proba'])
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        n_classes = len(self.classes)
        label_encoder = LabelEncoder().fit(self.classes)
        pos_class_indices = label_encoder.transform(self.estimator.classes_)
        proba = np.zeros((_num_samples(X), n_classes))
        for class_idx, this_pred, calibrator in zip(pos_class_indices, predictions.T, self.calibrators):
            if n_classes == 2:
                class_idx += 1
            proba[:, class_idx] = calibrator.predict(this_pred)
        if n_classes == 2:
            proba[:, 0] = 1.0 - proba[:, 1]
        else:
            denominator = np.sum(proba, axis=1)[:, np.newaxis]
            uniform_proba = np.full_like(proba, 1 / n_classes)
            proba = np.divide(proba, denominator, out=uniform_proba, where=denominator != 0)
        proba[(1.0 < proba) & (proba <= 1.0 + 1e-05)] = 1.0
        return proba