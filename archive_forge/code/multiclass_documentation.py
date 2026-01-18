import warnings
from collections.abc import Sequence
from itertools import chain
import numpy as np
from scipy.sparse import issparse
from ..utils._array_api import get_namespace
from ..utils.fixes import VisibleDeprecationWarning
from .validation import _assert_all_finite, check_array
Compute a continuous, tie-breaking OvR decision function from OvO.

    It is important to include a continuous value, not only votes,
    to make computing AUC or calibration meaningful.

    Parameters
    ----------
    predictions : array-like of shape (n_samples, n_classifiers)
        Predicted classes for each binary classifier.

    confidences : array-like of shape (n_samples, n_classifiers)
        Decision functions or predicted probabilities for positive class
        for each binary classifier.

    n_classes : int
        Number of classes. n_classifiers must be
        ``n_classes * (n_classes - 1 ) / 2``.
    