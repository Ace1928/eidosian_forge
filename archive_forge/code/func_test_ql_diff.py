from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
@pytest.mark.parametrize('family', [families.Gaussian, families.Binomial, families.Poisson])
def test_ql_diff(family):
    fam = family()
    y, x1, x2, g = simple_qic_data(family)
    model1 = gee.GEE(y, x1, family=fam, groups=g)
    result1 = model1.fit(ddof_scale=0)
    mean1 = result1.fittedvalues
    model2 = gee.GEE(y, x2, family=fam, groups=g)
    result2 = model2.fit(ddof_scale=0)
    mean2 = result2.fittedvalues
    if family is families.Gaussian:
        qldiff = 0
    elif family is families.Binomial:
        qldiff = np.sum(y * np.log(mean1 / (1 - mean1)) + np.log(1 - mean1))
        qldiff -= np.sum(y * np.log(mean2 / (1 - mean2)) + np.log(1 - mean2))
    elif family is families.Poisson:
        qldiff = np.sum(y * np.log(mean1) - mean1) - np.sum(y * np.log(mean2) - mean2)
    else:
        raise ValueError('unknown family')
    qle1, _, _ = model1.qic(result1.params, result1.scale, result1.cov_params())
    qle2, _, _ = model2.qic(result2.params, result2.scale, result2.cov_params())
    assert_allclose(qle1 - qle2, qldiff, rtol=1e-05, atol=1e-05)