from abc import ABCMeta, abstractmethod
from statsmodels.compat.python import with_metaclass
import itertools
import numpy as np
from statsmodels.gam.smooth_basis import (GenericSmoothers,
class MultivariateGAMCVPath:
    """k-fold cross-validation for GAM

    Warning: The API of this class is preliminary and will change.

    Parameters
    ----------
    smoother : additive smoother instance
    alphas : list of iteratables
        list of alpha for smooths. The product space will be used as alpha
        grid for cross-validation
    gam : model class
        model class for creating a model with k-fole training data
    cost : function
        cost function for the prediction error
    endog : ndarray
        dependent (response) variable of the model
    cv_iterator : instance of cross-validation iterator
    """

    def __init__(self, smoother, alphas, gam, cost, endog, exog, cv_iterator):
        self.cost = cost
        self.smoother = smoother
        self.gam = gam
        self.alphas = alphas
        self.alphas_grid = list(itertools.product(*self.alphas))
        self.endog = endog
        self.exog = exog
        self.cv_iterator = cv_iterator
        self.cv_error = np.zeros(shape=len(self.alphas_grid))
        self.cv_std = np.zeros(shape=len(self.alphas_grid))
        self.alpha_cv = None

    def fit(self, **kwargs):
        for i, alphas_i in enumerate(self.alphas_grid):
            gam_cv = MultivariateGAMCV(smoother=self.smoother, alphas=alphas_i, gam=self.gam, cost=self.cost, endog=self.endog, exog=self.exog, cv_iterator=self.cv_iterator)
            cv_err = gam_cv.fit(**kwargs)
            self.cv_error[i] = cv_err.mean()
            self.cv_std[i] = cv_err.std()
        self.alpha_cv = self.alphas_grid[np.argmin(self.cv_error)]
        return self