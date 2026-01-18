import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def weight_intercept_raw(self, coef, X):
    """Helper function to get coefficients, intercept and raw_prediction.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        raw_prediction : ndarray of shape (n_samples,) or             (n_samples, n_classes)
        """
    weights, intercept = self.weight_intercept(coef)
    if not self.base_loss.is_multiclass:
        raw_prediction = X @ weights + intercept
    else:
        raw_prediction = X @ weights.T + intercept
    return (weights, intercept, raw_prediction)