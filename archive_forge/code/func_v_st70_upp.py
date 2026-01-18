from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def v_st70_upp(stat, nobs):
    mod_factor = np.sqrt(nobs) + 0.155 + 0.24 / np.sqrt(nobs)
    stat_modified = stat * mod_factor
    zsqu = stat_modified ** 2
    pval = (8 * zsqu - 2) * np.exp(-2 * zsqu)
    digits = np.sum(stat > np.array([1.06, 1.06, 1.26]))
    return (stat_modified, pval, digits)