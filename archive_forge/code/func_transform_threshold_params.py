import warnings
from statsmodels.compat.pandas import Appender
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from scipy import stats
from statsmodels.base.model import (
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import cache_readonly
def transform_threshold_params(self, params):
    """transformation of the parameters in the optimization

        Parameters
        ----------
        params : nd_array
            Contains (exog_coef, transformed_thresholds) where exog_coef are
            the coefficient for the explanatory variables in the linear term,
            transformed threshold or cutoff points. The first, lowest threshold
            is unchanged, all other thresholds are in terms of exponentiated
            increments.

        Returns
        -------
        thresh : nd_array
            Thresh are the thresholds or cutoff constants for the intervals.

        """
    th_params = params[-(self.k_levels - 1):]
    thresh = np.concatenate((th_params[:1], np.exp(th_params[1:]))).cumsum()
    thresh = np.concatenate(([-np.inf], thresh, [np.inf]))
    return thresh