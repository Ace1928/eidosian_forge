from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
@Substitution(model_type='Ordinary', model='OLS', parameters=common_params, extra_parameters=extra_parameters)
@Appender(_doc)
class RollingOLS(RollingWLS):

    def __init__(self, endog, exog, window=None, *, min_nobs=None, missing='drop', expanding=False):
        super().__init__(endog, exog, window, weights=None, min_nobs=min_nobs, missing=missing, expanding=expanding)