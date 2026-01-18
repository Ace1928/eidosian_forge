import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
def test_right_censored_ci_nans(self):
    times, died = (self.t1, self.d1)
    sample = stats.CensoredData.right_censored(times, np.logical_not(died))
    res = stats.ecdf(sample)
    x = [37, 47, 56, 77, 80, 81]
    flo = [np.nan, 0, 0, 0.052701464070711, 0.33761112623179, np.nan]
    fup = [np.nan, 0.35417230377, 0.5500569798, 0.9472985359, 1.0, np.nan]
    i = np.searchsorted(res.cdf.quantiles, x)
    message = 'The confidence interval is undefined at some observations'
    with pytest.warns(RuntimeWarning, match=message):
        ci = res.cdf.confidence_interval()
    assert_allclose(ci.low.probabilities[i][1:], flo[1:])
    assert_allclose(ci.high.probabilities[i][1:], fup[1:])
    flo = [np.nan, 0.64582769623, 0.449943020228, 0.05270146407, 0, np.nan]
    fup = [np.nan, 1.0, 1.0, 0.947298535929289, 0.66238887376821, np.nan]
    i = np.searchsorted(res.cdf.quantiles, x)
    with pytest.warns(RuntimeWarning, match=message):
        ci = res.sf.confidence_interval()
    assert_allclose(ci.low.probabilities[i][1:], flo[1:])
    assert_allclose(ci.high.probabilities[i][1:], fup[1:])
    low = [1.0, 1.0, 0.6458276962323382, 0.44994302022779326, 0.44994302022779326, 0.44994302022779326, 0.44994302022779326, 0.05270146407071086, 0.0, np.nan]
    high = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9472985359292891, 0.6623888737682101, np.nan]
    assert_allclose(ci.low.probabilities, low)
    assert_allclose(ci.high.probabilities, high)
    with pytest.warns(RuntimeWarning, match=message):
        ci = res.sf.confidence_interval(method='log-log')
    low = [np.nan, np.nan, 0.3870000140320252, 0.3148071137055191, 0.3148071137055191, 0.3148071137055191, 0.3148071137055191, 0.08048821148507734, 0.01049958986680601, np.nan]
    high = [np.nan, np.nan, 0.981392965878966, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.9308983170906275, 0.8263946341076415, 0.6558775085110887, np.nan]
    assert_allclose(ci.low.probabilities, low)
    assert_allclose(ci.high.probabilities, high)