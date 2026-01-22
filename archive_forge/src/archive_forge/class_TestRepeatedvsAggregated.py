import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
class TestRepeatedvsAggregated(CheckWeight):

    @classmethod
    def setup_class(cls):
        np.random.seed(4321)
        n = 100
        p = 5
        exog = np.empty((n, p))
        exog[:, 0] = 1
        exog[:, 1] = np.random.randint(low=-5, high=5, size=n)
        x = np.repeat(np.array([1, 2, 3, 4]), n / 4)
        exog[:, 2:] = get_dummies(x)
        beta = np.array([-1, 0.1, -0.05, 0.2, 0.35])
        lin_pred = (exog * beta).sum(axis=1)
        family = sm.families.Poisson
        link = sm.families.links.Log
        endog = gen_endog(lin_pred, family, link)
        mod1 = sm.GLM(endog, exog, family=family(link=link()))
        cls.res1 = mod1.fit()
        agg = pd.DataFrame(exog)
        agg['endog'] = endog
        agg_endog = agg.groupby([0, 1, 2, 3, 4]).sum()[['endog']]
        agg_wt = agg.groupby([0, 1, 2, 3, 4]).count()[['endog']]
        agg_exog = np.array(agg_endog.index.tolist())
        agg_wt = agg_wt['endog']
        agg_endog = agg_endog['endog']
        mod2 = sm.GLM(agg_endog, agg_exog, family=family(link=link()), exposure=agg_wt)
        cls.res2 = mod2.fit()