import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_predict_custom_index():
    np.random.seed(328423)
    endog = pd.DataFrame(np.random.normal(size=(50, 2)))
    mod = dynamic_factor.DynamicFactor(endog, k_factors=1, factor_order=1)
    res = mod.smooth(mod.start_params)
    out = res.predict(start=1, end=1, index=['a'])
    assert_equal(out.index.equals(pd.Index(['a'])), True)