import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
@pytest.mark.parametrize('case', [(t1, d1, r1), (t2, d2, r2), (t3, d3, r3), (t4, d4, r4), (t5, d5, r5)])
def test_right_censored_against_examples(self, case):
    times, died, ref = case
    sample = stats.CensoredData.right_censored(times, np.logical_not(died))
    res = stats.ecdf(sample)
    assert_allclose(res.sf.probabilities, ref, atol=0.001)
    assert_equal(res.sf.quantiles, np.sort(np.unique(times)))
    res = _kaplan_meier_reference(times, np.logical_not(died))
    assert_equal(res[0], np.sort(np.unique(times)))
    assert_allclose(res[1], ref, atol=0.001)