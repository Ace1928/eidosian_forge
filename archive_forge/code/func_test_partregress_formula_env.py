import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@pytest.mark.matplotlib
def test_partregress_formula_env():

    @np.vectorize
    def lg(x):
        return np.log10(x) if x > 0 else 0
    df = DataFrame(dict(a=np.random.random(size=10), b=np.random.random(size=10), c=np.random.random(size=10)))
    sm.graphics.plot_partregress('a', 'lg(b)', ['c'], obs_labels=False, data=df, eval_env=1)
    sm.graphics.plot_partregress('a', 'lg(b)', ['c'], obs_labels=False, data=df)