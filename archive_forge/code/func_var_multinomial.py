import numpy as np
from scipy import stats
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.effect_size import _noncentrality_chisquare
def var_multinomial(probs):
    """variance of multinomial distribution

    var = probs * (1 - probs)

    """
    var = probs * (1 - probs)
    return var