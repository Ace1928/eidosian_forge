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
@pytest.mark.parametrize('family', [families.Gaussian, families.Poisson])
def test_ql_known(family):
    fam = family()
    y, x1, x2, g = simple_qic_data(family)
    model1 = gee.GEE(y, x1, family=fam, groups=g)
    result1 = model1.fit(ddof_scale=0)
    mean1 = result1.fittedvalues
    model2 = gee.GEE(y, x2, family=fam, groups=g)
    result2 = model2.fit(ddof_scale=0)
    mean2 = result2.fittedvalues
    if family is families.Gaussian:
        ql1 = -len(y) / 2.0
        ql2 = -len(y) / 2.0
    elif family is families.Poisson:
        c = np.zeros_like(y)
        ii = y > 0
        c[ii] = y[ii] * np.log(y[ii]) - y[ii]
        ql1 = np.sum(y * np.log(mean1) - mean1 - c)
        ql2 = np.sum(y * np.log(mean2) - mean2 - c)
    else:
        raise ValueError('Unknown family')
    qle1 = model1.qic(result1.params, result1.scale, result1.cov_params())
    qle2 = model2.qic(result2.params, result2.scale, result2.cov_params())
    assert_allclose(ql1, qle1[0], rtol=0.0001)
    assert_allclose(ql2, qle2[0], rtol=0.0001)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qler1 = result1.qic()
        qler2 = result2.qic()
    assert_allclose(qler1, qle1[1:], rtol=1e-05)
    assert_allclose(qler2, qle2[1:], rtol=1e-05)