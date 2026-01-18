from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
@pytest.mark.slow
def test_vcomp_formula(self):
    np.random.seed(6241)
    n = 800
    exog = np.random.normal(size=(n, 2))
    exog[:, 0] = 1
    ex_vc = []
    groups = np.kron(np.arange(n / 4), np.ones(4))
    errors = 0
    exog_re = np.random.normal(size=(n, 2))
    slopes = np.random.normal(size=(n // 4, 2))
    slopes = np.kron(slopes, np.ones((4, 1))) * exog_re
    errors += slopes.sum(1)
    ex_vc = np.random.normal(size=(n, 4))
    slopes = np.random.normal(size=(n // 4, 4))
    slopes[:, 2:] *= 2
    slopes = np.kron(slopes, np.ones((4, 1))) * ex_vc
    errors += slopes.sum(1)
    errors += np.random.normal(size=n)
    endog = exog.sum(1) + errors
    exog_vc = {'a': {}, 'b': {}}
    for k, group in enumerate(range(int(n / 4))):
        ix = np.flatnonzero(groups == group)
        exog_vc['a'][group] = ex_vc[ix, 0:2]
        exog_vc['b'][group] = ex_vc[ix, 2:]
    with pytest.warns(UserWarning, match='Using deprecated variance'):
        model1 = MixedLM(endog, exog, groups, exog_re=exog_re, exog_vc=exog_vc)
    result1 = model1.fit()
    df = pd.DataFrame(exog[:, 1:], columns=['x1'])
    df['y'] = endog
    df['re1'] = exog_re[:, 0]
    df['re2'] = exog_re[:, 1]
    df['vc1'] = ex_vc[:, 0]
    df['vc2'] = ex_vc[:, 1]
    df['vc3'] = ex_vc[:, 2]
    df['vc4'] = ex_vc[:, 3]
    vc_formula = {'a': '0 + vc1 + vc2', 'b': '0 + vc3 + vc4'}
    model2 = MixedLM.from_formula('y ~ x1', groups=groups, re_formula='0 + re1 + re2', vc_formula=vc_formula, data=df)
    result2 = model2.fit()
    assert_allclose(result1.fe_params, result2.fe_params, rtol=1e-08)
    assert_allclose(result1.cov_re, result2.cov_re, rtol=1e-08)
    assert_allclose(result1.vcomp, result2.vcomp, rtol=1e-08)
    assert_allclose(result1.params, result2.params, rtol=1e-08)
    assert_allclose(result1.bse, result2.bse, rtol=1e-08)