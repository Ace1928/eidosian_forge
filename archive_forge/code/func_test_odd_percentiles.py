import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_odd_percentiles(df):
    percentiles = np.linspace(7.0, 93.0, 13)
    res = Description(df, percentiles=percentiles)
    stats = ['nobs', 'missing', 'mean', 'std_err', 'upper_ci', 'lower_ci', 'std', 'iqr', 'iqr_normal', 'mad', 'mad_normal', 'coef_var', 'range', 'max', 'min', 'skew', 'kurtosis', 'jarque_bera', 'jarque_bera_pval', 'mode', 'mode_freq', 'median', 'distinct', 'top_1', 'top_2', 'top_3', 'top_4', 'top_5', 'freq_1', 'freq_2', 'freq_3', 'freq_4', 'freq_5', '7.0%', '14.1%', '21.3%', '28.5%', '35.6%', '42.8%', '50.0%', '57.1%', '64.3%', '71.5%', '78.6%', '85.8%', '93.0%']
    assert_equal(res.frame.index.tolist(), stats)