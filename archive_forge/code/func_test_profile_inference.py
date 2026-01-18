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
@pytest.mark.smoke
def test_profile_inference(self):
    np.random.seed(9814)
    k_fe = 2
    gsize = 3
    n_grp = 100
    exog = np.random.normal(size=(n_grp * gsize, k_fe))
    exog_re = np.ones((n_grp * gsize, 1))
    groups = np.kron(np.arange(n_grp), np.ones(gsize))
    vca = np.random.normal(size=n_grp * gsize)
    vcb = np.random.normal(size=n_grp * gsize)
    errors = 0
    g_errors = np.kron(np.random.normal(size=100), np.ones(gsize))
    errors += g_errors + exog_re[:, 0]
    rc = np.random.normal(size=n_grp)
    errors += np.kron(rc, np.ones(gsize)) * vca
    rc = np.random.normal(size=n_grp)
    errors += np.kron(rc, np.ones(gsize)) * vcb
    errors += np.random.normal(size=n_grp * gsize)
    endog = exog.sum(1) + errors
    vc = {'a': {}, 'b': {}}
    for k in range(n_grp):
        ii = np.flatnonzero(groups == k)
        vc['a'][k] = vca[ii][:, None]
        vc['b'][k] = vcb[ii][:, None]
    with pytest.warns(UserWarning, match='Using deprecated variance'):
        rslt = MixedLM(endog, exog, groups=groups, exog_re=exog_re, exog_vc=vc).fit()
    rslt.profile_re(0, vtype='re', dist_low=1, num_low=3, dist_high=1, num_high=3)
    rslt.profile_re('b', vtype='vc', dist_low=0.5, num_low=3, dist_high=0.5, num_high=3)