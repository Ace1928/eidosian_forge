from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def usqu_st70_upp(stat, nobs):
    nobsinv = 1.0 / nobs
    stat_modified = stat - 0.1 * nobsinv + 0.1 * nobsinv ** 2
    stat_modified *= 1 + 0.8 * nobsinv
    pval = 2 * np.exp(-2 * stat_modified * np.pi ** 2)
    digits = np.sum(stat > np.array([0.29, 0.29, 0.34]))
    return (stat_modified, pval, digits)