from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
class Margins:
    """
    Mostly a do nothing class. Lays out the methods expected of a sub-class.

    This is just a sketch of what we may want out of a general margins class.
    I (SS) need to look at details of other models.
    """

    def __init__(self, results, get_margeff, derivative, dist=None, margeff_args=()):
        self._cache = {}
        self.results = results
        self.dist = dist
        self.get_margeff(margeff_args)

    def _reset(self):
        self._cache = {}

    def get_margeff(self, *args, **kwargs):
        self._reset()
        self.margeff = self.get_margeff(*args)

    @cache_readonly
    def tvalues(self):
        raise NotImplementedError

    @cache_readonly
    def cov_margins(self):
        raise NotImplementedError

    @cache_readonly
    def margins_se(self):
        raise NotImplementedError

    def summary_frame(self):
        raise NotImplementedError

    @cache_readonly
    def pvalues(self):
        raise NotImplementedError

    def conf_int(self, alpha=0.05):
        raise NotImplementedError

    def summary(self, alpha=0.05):
        raise NotImplementedError