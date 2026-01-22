from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
class NewNorm:
    """just a holder for modified distributions
    """

    def fit_vec(self, x, axis=0):
        return (x.mean(axis), x.std(axis))

    def cdf(self, x, args):
        return distributions.norm.cdf(x, loc=args[0], scale=args[1])

    def rvs(self, args, size):
        loc = args[0]
        scale = args[1]
        return loc + scale * distributions.norm.rvs(size=size)