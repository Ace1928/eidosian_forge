import pytest
import warnings
import numpy as np
from numpy import arange
from numpy.testing import assert_allclose, assert_equal
from scipy import stats
import statsmodels.stats.rates as smr
from statsmodels.stats.rates import (
class TestMethodsCompar1samp:

    @pytest.mark.parametrize('meth', method_names_poisson_1samp['test'])
    def test_test(self, meth):
        count1, n1 = (60, 514.775)
        tst = smr.test_poisson(count1, n1, method=meth, value=0.1, alternative='two-sided')
        assert_allclose(tst.pvalue, 0.25, rtol=0.1)

    @pytest.mark.parametrize('meth', method_names_poisson_1samp['confint'])
    def test_confint(self, meth):
        count1, n1 = (60, 514.775)
        ci = confint_poisson(count1, n1, method=meth, alpha=0.05)
        assert_allclose(ci, [0.089, 0.158], rtol=0.1)