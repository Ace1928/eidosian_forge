import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def _invert_test_confint(count, nobs, alpha=0.05, method='midp-c', method_start='exact-c'):
    """invert hypothesis test to get confidence interval
    """

    def func(r):
        v = (test_poisson(count, nobs, value=r, method=method)[1] - alpha) ** 2
        return v
    ci = confint_poisson(count, nobs, method=method_start)
    low = optimize.fmin(func, ci[0], xtol=1e-08, disp=False)
    upp = optimize.fmin(func, ci[1], xtol=1e-08, disp=False)
    assert np.size(low) == 1
    return (low[0], upp[0])