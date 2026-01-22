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
def _linpred(self, params, exog=None, offset=None):
    """Linear prediction of latent variable `x b + offset`.

        Parameters
        ----------
        params : ndarray
            Parameters for the model, (exog_coef, transformed_thresholds)
        exog : array_like, optional
            Design / exogenous data. Is exog is None, model exog is used.
        offset : array_like, optional
            Offset is added to the linear prediction with coefficient
            equal to 1. If offset is not provided and exog
            is None, uses the model's offset if present.  If not, uses
            0 as the default value.

        Returns
        -------
        linear : ndarray
            1-dim linear prediction given by exog times linear params plus
            offset. This is the prediction for the underlying latent variable.
            If exog and offset are None, then the predicted values are zero.

        """
    if exog is None:
        exog = self.exog
        if offset is None:
            offset = self.offset
    elif offset is None:
        offset = 0
    if offset is not None:
        offset = np.asarray(offset)
    if exog is not None:
        _exog = np.asarray(exog)
        _params = np.asarray(params)
        linpred = _exog.dot(_params[:-(self.k_levels - 1)])
    else:
        linpred = np.zeros(self.nobs)
    if offset is not None:
        linpred += offset
    return linpred