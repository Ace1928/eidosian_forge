import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
def test_right_censored_ci_example_5(self):
    times, died = (self.t5, self.d5)
    sample = stats.CensoredData.right_censored(times, np.logical_not(died))
    res = stats.ecdf(sample)
    lower = np.array([0.66639, 0.624174, 0.456179, 0.287822, 0.287822, 0.287822, 0.128489, 0.030957, 0.030957, 0.030957])
    upper = np.array([0.991983, 0.970995, 0.87378, 0.739467, 0.739467, 0.739467, 0.603133, 0.430365, 0.430365, 0.430365])
    sf_ci = res.sf.confidence_interval(method='log-log')
    cdf_ci = res.cdf.confidence_interval(method='log-log')
    assert_allclose(sf_ci.low.probabilities, lower, atol=1e-05)
    assert_allclose(sf_ci.high.probabilities, upper, atol=1e-05)
    assert_allclose(cdf_ci.low.probabilities, 1 - upper, atol=1e-05)
    assert_allclose(cdf_ci.high.probabilities, 1 - lower, atol=1e-05)
    low = [0.7436674840686117, 0.6858233228919625, 0.5059683565148012, 0.32913131413336727, 0.32913131413336727, 0.32913131413336727, 0.15986912028781664, 0.04499539918147757, 0.04499539918147757, 0.04499539918147757]
    high = [0.9890291867238429, 0.9638835422144144, 0.8560366823086629, 0.713016764397845, 0.713016764397845, 0.713016764397845, 0.5678602982997164, 0.3887616766886558, 0.3887616766886558, 0.3887616766886558]
    sf_ci = res.sf.confidence_interval(method='log-log', confidence_level=0.9)
    assert_allclose(sf_ci.low.probabilities, low)
    assert_allclose(sf_ci.high.probabilities, high)
    low = [0.8556383113628162, 0.7670478794850761, 0.5485720663578469, 0.3441515412527123, 0.3441515412527123, 0.3441515412527123, 0.1449184105424544, 0.0, 0.0, 0.0]
    high = [1.0, 1.0, 0.8958723780865975, 0.739181792080621, 0.739181792080621, 0.739181792080621, 0.5773038116797676, 0.364227025459672, 0.364227025459672, 0.364227025459672]
    sf_ci = res.sf.confidence_interval(confidence_level=0.9)
    assert_allclose(sf_ci.low.probabilities, low)
    assert_allclose(sf_ci.high.probabilities, high)