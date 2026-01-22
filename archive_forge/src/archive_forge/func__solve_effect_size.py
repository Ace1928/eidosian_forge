import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def _solve_effect_size(self, effect_size=None, nobs=None, alpha=None, power=None, k_groups=2):
    """experimental, test failure in solve_power for effect_size
        """

    def func(x):
        effect_size = x
        return self._power_identity(effect_size=effect_size, nobs=nobs, alpha=alpha, k_groups=k_groups, power=power)
    val, r = optimize.brentq(func, 1e-08, 1 - 1e-08, full_output=True)
    if not r.converged:
        print(r)
    return val