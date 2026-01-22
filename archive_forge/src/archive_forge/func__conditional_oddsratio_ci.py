import numpy as np
from scipy.special import ndtri
from scipy.optimize import brentq
from ._discrete_distns import nchypergeom_fisher
from ._common import ConfidenceInterval
def _conditional_oddsratio_ci(table, confidence_level=0.95, alternative='two-sided'):
    """
    Conditional exact confidence interval for the odds ratio.
    """
    if alternative == 'two-sided':
        alpha = 0.5 * (1 - confidence_level)
        lower = _ci_lower(table, alpha)
        upper = _ci_upper(table, alpha)
    elif alternative == 'less':
        lower = 0.0
        upper = _ci_upper(table, 1 - confidence_level)
    else:
        lower = _ci_lower(table, 1 - confidence_level)
        upper = np.inf
    return (lower, upper)