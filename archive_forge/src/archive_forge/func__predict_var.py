import numpy as np
from scipy.special import gammaln as lgamma
import patsy
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
from statsmodels.base.model import (
from statsmodels.genmod import families
def _predict_var(self, params, exog=None, exog_precision=None):
    """predict values for conditional variance V(endog | exog)

        Parameters
        ----------
        params : array_like
            The model parameters.
        exog : array_like
            Array of predictor variables for mean.
        exog_precision : array_like
            Array of predictor variables for precision.

        Returns
        -------
        Predicted conditional variance.
        """
    mean = self.predict(params, exog=exog)
    precision = self._predict_precision(params, exog_precision=exog_precision)
    var_endog = mean * (1 - mean) / (1 + precision)
    return var_endog