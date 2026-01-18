from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def wsqu_st70_upp(stat, nobs):
    nobsinv = 1.0 / nobs
    stat_modified = (stat - 0.4 * nobsinv + 0.6 * nobsinv ** 2) * (1 + nobsinv)
    pval = 0.05 * np.exp(2.79 - 6 * stat_modified)
    digits = np.nan
    return (stat_modified, pval, digits)