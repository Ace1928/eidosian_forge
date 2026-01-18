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
def transform_reverse_threshold_params(self, params):
    """obtain transformed thresholds from original thresholds or cutoffs

        Parameters
        ----------
        params : ndarray
            Threshold values, cutoff constants for choice intervals, which
            need to be monotonically increasing.

        Returns
        -------
        thresh_params : ndarrray
            Transformed threshold parameter.
            The first, lowest threshold is unchanged, all other thresholds are
            in terms of exponentiated increments.
            Transformed parameters can be any real number without restrictions.

        """
    thresh_params = np.concatenate((params[:1], np.log(np.diff(params[:-1]))))
    return thresh_params