import numpy as np
from scipy import sparse
from ..utils.extmath import squared_norm
def weight_intercept(self, coef):
    """Helper function to get coefficients and intercept.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        """
    if not self.base_loss.is_multiclass:
        if self.fit_intercept:
            intercept = coef[-1]
            weights = coef[:-1]
        else:
            intercept = 0.0
            weights = coef
    else:
        if coef.ndim == 1:
            weights = coef.reshape((self.base_loss.n_classes, -1), order='F')
        else:
            weights = coef
        if self.fit_intercept:
            intercept = weights[:, -1]
            weights = weights[:, :-1]
        else:
            intercept = 0.0
    return (weights, intercept)