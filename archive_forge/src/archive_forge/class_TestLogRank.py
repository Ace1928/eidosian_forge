import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
class TestLogRank:

    @pytest.mark.parametrize('x, y, statistic, pvalue', [([[8, 12, 26, 14, 21, 27], [8, 32, 20, 40]], [[33, 28, 41], [48, 48, 25, 37, 48, 25, 43]], 6.91598157449, [0.008542873404, 0.9957285632979385, 0.004271436702061537]), ([[19, 6, 5, 4], [20, 19, 17, 14]], [[16, 21, 7], [21, 15, 18, 18, 5]], 0.835004855038, [0.3608293039, 0.8195853480676912, 0.1804146519323088]), ([[6, 13, 21, 30, 37, 38, 49, 50, 63, 79, 86, 98, 202, 219], [31, 47, 80, 82, 82, 149]], [[10, 10, 12, 13, 14, 15, 16, 17, 18, 20, 24, 24, 25, 28, 30, 33, 35, 37, 40, 40, 46, 48, 76, 81, 82, 91, 112, 181], [34, 40, 70]], 7.49659416854, [0.006181578637, 0.003090789318730882, 0.9969092106812691])])
    def test_log_rank(self, x, y, statistic, pvalue):
        x = stats.CensoredData(uncensored=x[0], right=x[1])
        y = stats.CensoredData(uncensored=y[0], right=y[1])
        for i, alternative in enumerate(['two-sided', 'less', 'greater']):
            res = stats.logrank(x=x, y=y, alternative=alternative)
            assert_allclose(res.statistic ** 2, statistic, atol=1e-10)
            assert_allclose(res.pvalue, pvalue[i], atol=1e-10)

    def test_raises(self):
        sample = stats.CensoredData([1, 2])
        msg = '`y` must be'
        with pytest.raises(ValueError, match=msg):
            stats.logrank(x=sample, y=[[1, 2]])
        msg = '`x` must be'
        with pytest.raises(ValueError, match=msg):
            stats.logrank(x=[[1, 2]], y=sample)