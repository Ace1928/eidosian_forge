import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
def test_rate_poisson_r():
    count, nobs = (15, 400)
    pv2 = 0.313026269279486
    ci2 = (0.0209884653319583, 0.0618505471787146)
    rt = smr.test_poisson(count, nobs, value=0.05, method='exact-c')
    ci = confint_poisson(count, nobs, method='exact-c')
    assert_allclose(rt.pvalue, pv2, rtol=1e-12)
    assert_allclose(ci, ci2, rtol=1e-12)
    pv2 = 0.263552477282973
    ci2 = (0.0227264749053794, 0.0618771721463559)
    rt = smr.test_poisson(count, nobs, value=0.05, method='score')
    ci = confint_poisson(count, nobs, method='score')
    assert_allclose(rt.pvalue, pv2, rtol=1e-12)
    assert_allclose(ci, ci2, rtol=1e-12)
    ci2 = (0.0219234232268444, 0.0602898619930649)
    ci = confint_poisson(count, nobs, method='jeff')
    assert_allclose(ci, ci2, rtol=1e-12)
    ci2 = (0.0185227303217751, 0.0564772696782249)
    ci = confint_poisson(count, nobs, method='wald')
    assert_allclose(ci, ci2, rtol=1e-12)
    ci2 = (0.0243357599260795, 0.0604627555786095)
    ci = confint_poisson(count, nobs, method='midp-c')
    assert_allclose(ci[1], ci2[1], rtol=1e-05)