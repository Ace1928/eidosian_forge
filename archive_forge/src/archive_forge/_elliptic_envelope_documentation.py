from numbers import Real
import numpy as np
from ..base import OutlierMixin, _fit_context
from ..metrics import accuracy_score
from ..utils._param_validation import Interval
from ..utils.validation import check_is_fitted
from ._robust_covariance import MinCovDet
Return the mean accuracy on the given test data and labels.

        In multi-label classification, this is the subset accuracy
        which is a harsh metric since you require for each sample that
        each label set be correctly predicted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        