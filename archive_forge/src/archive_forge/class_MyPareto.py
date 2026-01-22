import numpy as np
from scipy import stats
from statsmodels.base.model import GenericLikelihoodModel
from numpy.testing import (assert_array_less, assert_almost_equal,
class MyPareto(GenericLikelihoodModel):
    """Maximum Likelihood Estimation pareto distribution

    first version: iid case, with constant parameters
    """

    def initialize(self):
        super().initialize()
        extra_params_names = ['shape', 'loc', 'scale']
        self._set_extra_params_names(extra_params_names)
        self.start_params = np.array([1.5, self.endog.min() - 1.5, 1.0])

    def pdf(self, x, b):
        return b * x ** (-b - 1)

    def loglike(self, params):
        return -self.nloglikeobs(params).sum(0)

    def nloglikeobs(self, params):
        if self.fixed_params is not None:
            params = self.expandparams(params)
        b = params[0]
        loc = params[1]
        scale = params[2]
        endog = self.endog
        x = (endog - loc) / scale
        logpdf = np.log(b) - (b + 1.0) * np.log(x)
        logpdf -= np.log(scale)
        logpdf[x < 1] = -10000
        return -logpdf