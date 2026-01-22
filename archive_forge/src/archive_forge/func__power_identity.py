import warnings
import numpy as np
from scipy import stats, optimize, special
from statsmodels.tools.rootfinding import brentq_expanding
def _power_identity(self, *args, **kwds):
    power_ = kwds.pop('power')
    return self.power(*args, **kwds) - power_