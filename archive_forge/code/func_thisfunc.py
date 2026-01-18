import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
def thisfunc(x):
    xn = (x - mu) / sig
    return totp(xn) * np.exp(-xn * xn / 2.0) / np.sqrt(2 * np.pi) / sig