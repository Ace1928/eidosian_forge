import copy
import warnings
import numpy as np
from numpy.testing import (assert_almost_equal, assert_allclose, assert_raises,
import pytest
import statsmodels.stats.power as smp
from statsmodels.stats.tests.test_weightstats import Holder
from statsmodels.tools.sm_exceptions import HypothesisTestWarning
def test_power_solver():
    nip = smp.NormalIndPower()
    es0 = 0.1
    pow_ = nip.solve_power(es0, nobs1=1600, alpha=0.01, power=None, ratio=1, alternative='larger')
    assert_almost_equal(pow_, 0.6921941124382421, decimal=5)
    es = nip.solve_power(None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative='larger')
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 2)
    nip.start_bqexp['effect_size'] = {'upp': -10, 'low': -20}
    nip.start_ttp['effect_size'] = 0.14
    es = nip.solve_power(None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative='larger')
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 3, err_msg=repr(nip.cache_fit_res))
    nip.start_ttp['effect_size'] = np.nan
    es = nip.solve_power(None, nobs1=1600, alpha=0.01, power=pow_, ratio=1, alternative='larger')
    assert_almost_equal(es, es0, decimal=4)
    assert_equal(nip.cache_fit_res[0], 1)
    assert_equal(len(nip.cache_fit_res), 4)
    es = nip.solve_power(nobs1=1600, alpha=0.01, effect_size=0, power=None)
    assert_almost_equal(es, 0.01)
    assert_raises(ValueError, nip.solve_power, None, nobs1=1600, alpha=0.01, power=0.005, ratio=1, alternative='larger')
    with pytest.warns(HypothesisTestWarning):
        with pytest.raises(ValueError):
            nip.solve_power(nobs1=None, effect_size=0, alpha=0.01, power=0.005, ratio=1, alternative='larger')