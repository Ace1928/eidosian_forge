import numpy as np
from statsmodels.regression.linear_model import OLS, GLS, WLS

        Perform an iterative two-step procedure to estimate a WLS model.

        The model is assumed to have heteroskedastic errors.
        The variance is estimated by OLS regression of the link transformed
        squared residuals on Z, i.e.::

           link(sigma_i) = x_i*gamma.

        Parameters
        ----------
        maxiter : int, optional
            the number of iterations

        Notes
        -----
        maxiter=1: returns the estimated based on given weights
        maxiter=2: performs a second estimation with the updated weights,
                   this is 2-step estimation
        maxiter>2: iteratively estimate and update the weights

        TODO: possible extension stop iteration if change in parameter
            estimates is smaller than x_tol

        Repeated calls to fit_iterative, will do one redundant pinv_wexog
        calculation. Calling fit_iterative(maxiter) ones does not do any
        redundant recalculations (whitening or calculating pinv_wexog).
        