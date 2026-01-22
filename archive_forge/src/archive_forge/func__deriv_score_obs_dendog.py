import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _deriv_score_obs_dendog(self, params):
    """derivative of score_obs w.r.t. endog

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        derivative : ndarray_2d
            The derivative of the score_obs with respect to endog.
        """
    from statsmodels.tools.numdiff import _approx_fprime_cs_scalar

    def f(y):
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        sf = self.score_factor(params, endog=y)
        return np.column_stack(sf)
    dsf = _approx_fprime_cs_scalar(self.endog[:, None], f)
    d1 = dsf[:, :1] * self.exog
    d2 = dsf[:, 1:2] * self.exog_precision
    return np.column_stack((d1, d2))