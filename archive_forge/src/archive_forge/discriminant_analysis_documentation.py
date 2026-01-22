import warnings
from numbers import Integral, Real
import numpy as np
import scipy.linalg
from scipy import linalg
from .base import (
from .covariance import empirical_covariance, ledoit_wolf, shrunk_covariance
from .linear_model._base import LinearClassifierMixin
from .preprocessing import StandardScaler
from .utils._array_api import _expit, device, get_namespace, size
from .utils._param_validation import HasMethods, Interval, StrOptions
from .utils.extmath import softmax
from .utils.multiclass import check_classification_targets, unique_labels
from .utils.validation import check_is_fitted
Return log of posterior probabilities of classification.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Array of samples/test vectors.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Posterior log-probabilities of classification per class.
        