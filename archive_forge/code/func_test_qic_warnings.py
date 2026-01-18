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
def test_qic_warnings():
    with pytest.warns(UserWarning):
        fam = families.Gaussian()
        y, x1, _, g = simple_qic_data(fam)
        model = gee.GEE(y, x1, family=fam, groups=g)
        result = model.fit()
        result.qic()