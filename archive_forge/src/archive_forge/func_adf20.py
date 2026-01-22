from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.sandbox.tools.mctools import StatTestMC
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
def adf20(x):
    return adfuller(x, 2, regression='n', autolag=None)