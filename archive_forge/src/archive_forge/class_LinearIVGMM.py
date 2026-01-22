from statsmodels.compat.python import lrange
import numpy as np
from scipy import optimize, stats
from statsmodels.tools.numdiff import approx_fprime
from statsmodels.base.model import (Model,
from statsmodels.regression.linear_model import (OLS, RegressionResults,
import statsmodels.stats.sandwich_covariance as smcov
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import _ensure_2d
class LinearIVGMM(IVGMM):
    """class for linear instrumental variables models estimated with GMM

    Uses closed form expression instead of nonlinear optimizers for each step
    of the iterative GMM.

    The model is assumed to have the following moment condition

        E( z * (y - x beta)) = 0

    Where `y` is the dependent endogenous variable, `x` are the explanatory
    variables and `z` are the instruments. Variables in `x` that are exogenous
    need also be included in `z`.

    Notation Warning: our name `exog` stands for the explanatory variables,
    and includes both exogenous and explanatory variables that are endogenous,
    i.e. included endogenous variables

    Parameters
    ----------
    endog : array_like
        dependent endogenous variable
    exog : array_like
        explanatory, right hand side variables, including explanatory variables
        that are endogenous
    instrument : array_like
        Instrumental variables, variables that are exogenous to the error
        in the linear model containing both included and excluded exogenous
        variables
    """

    def fitgmm(self, start, weights=None, optim_method=None, **kwds):
        """estimate parameters using GMM for linear model

        Uses closed form expression instead of nonlinear optimizers

        Parameters
        ----------
        start : not used
            starting values for minimization, not used, only for consistency
            of method signature
        weights : ndarray
            weighting matrix for moment conditions. If weights is None, then
            the identity matrix is used
        optim_method : not used,
            optimization method, not used, only for consistency of method
            signature
        **kwds : keyword arguments
            not used, will be silently ignored (for compatibility with generic)


        Returns
        -------
        paramest : ndarray
            estimated parameters

        """
        if weights is None:
            weights = self.start_weights(inv=False)
        y, x, z = (self.endog, self.exog, self.instrument)
        zTx = np.dot(z.T, x)
        zTy = np.dot(z.T, y)
        part0 = zTx.T.dot(weights)
        part1 = part0.dot(zTx)
        part2 = part0.dot(zTy)
        params = np.linalg.pinv(part1).dot(part2)
        return params

    def predict(self, params, exog=None):
        if exog is None:
            exog = self.exog
        return np.dot(exog, params)

    def gradient_momcond(self, params, **kwds):
        x, z = (self.exog, self.instrument)
        gradmoms = -np.dot(z.T, x) / self.nobs
        return gradmoms

    def score(self, params, weights, **kwds):
        x, z = (self.exog, self.instrument)
        nobs = z.shape[0]
        u = self.get_errors(params)
        score = -2 * np.dot(x.T, z).dot(weights.dot(np.dot(z.T, u)))
        score /= nobs * nobs
        return score