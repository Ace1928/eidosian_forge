from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def test_pd_offset_exposure(self):
    endog = pd.DataFrame({'F': [0.0, 0.0, 0.0, 0.0, 1.0]})
    exog = pd.DataFrame({'I': [1.0, 1.0, 1.0, 1.0, 1.0], 'C': [0.0, 1.0, 0.0, 1.0, 0.0]})
    exposure = pd.Series([1.0, 1, 1, 2, 1])
    offset = pd.Series([1, 1, 1, 2, 1])
    sm.Poisson(endog=endog, exog=exog, offset=offset).fit()
    inflations = ['logit', 'probit']
    for inflation in inflations:
        sm.ZeroInflatedPoisson(endog=endog, exog=exog['I'], exposure=exposure, inflation=inflation).fit()