import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_use_t(df):
    res = Description(df)
    res_t = Description(df, use_t=True)
    assert res_t.frame.a.lower_ci < res.frame.a.lower_ci
    assert res_t.frame.a.upper_ci > res.frame.a.upper_ci