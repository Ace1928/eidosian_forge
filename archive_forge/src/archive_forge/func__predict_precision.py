import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _predict_precision(self, params, exog_precision=None):
    """Predict values for precision function for given exog_precision.

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted precision.
        """
    if exog_precision is None:
        exog_precision = self.exog_precision
    k_mean = self.exog.shape[1]
    params_precision = params[k_mean:]
    linpred_prec = np.dot(exog_precision, params_precision)
    phi = self.link_precision.inverse(linpred_prec)
    return phi