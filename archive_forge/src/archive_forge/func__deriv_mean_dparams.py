import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _deriv_mean_dparams(self, params):
    """
        Derivative of the expected endog with respect to the parameters.

        not verified yet

        Parameters
        ----------
        params : ndarray
            parameter at which score is evaluated

        Returns
        -------
        The value of the derivative of the expected endog with respect
        to the parameter vector.
        """
    link = self.link
    lin_pred = self.predict(params, which='linear')
    idl = link.inverse_deriv(lin_pred)
    dmat = self.exog * idl[:, None]
    return np.column_stack((dmat, np.zeros(self.exog_precision.shape)))