import numpy as np
from statsmodels.robust import mad
from scipy.optimize import minimize_scalar
def _loglik_boxcox(self, x, bounds, options={'maxiter': 25}):
    """
        Taken from the Stata manual on Box-Cox regressions, where this is the
        special case of 'lhs only'. As an estimator for the variance, the
        sample variance is used, by means of the well-known formula.

        Parameters
        ----------
        x : array_like
        options : dict
            The options (as a dict) to be passed to the optimizer.
        """
    sum_x = np.sum(np.log(x))
    nobs = len(x)

    def optim(lmbda):
        y, lmbda = self.transform_boxcox(x, lmbda)
        return (1 - lmbda) * sum_x + nobs / 2.0 * np.log(np.var(y))
    res = minimize_scalar(optim, bounds=bounds, method='bounded', options=options)
    return res.x