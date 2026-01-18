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
def test_vcomp_3(self):
    np.random.seed(4279)
    x1 = np.random.normal(size=400)
    groups = np.kron(np.arange(100), np.ones(4))
    slopes = np.random.normal(size=100)
    slopes = np.kron(slopes, np.ones(4)) * x1
    y = slopes + np.random.normal(size=400)
    vc_fml = {'a': '0 + x1'}
    df = pd.DataFrame({'y': y, 'x1': x1, 'groups': groups})
    model = MixedLM.from_formula('y ~ 1', groups='groups', vc_formula=vc_fml, data=df)
    result = model.fit()
    result.summary()
    assert_allclose(result.resid.iloc[0:4], np.r_[-1.180753, 0.279966, 0.578576, -0.667916], rtol=0.001)
    assert_allclose(result.fittedvalues.iloc[0:4], np.r_[-0.101549, 0.028613, -0.224621, -0.126295], rtol=0.001)