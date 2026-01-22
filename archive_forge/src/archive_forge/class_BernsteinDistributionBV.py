import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.distributions.tools import (
class BernsteinDistributionBV(BernsteinDistribution):

    def cdf(self, x):
        cdf_ = _eval_bernstein_2d(x, self.cdf_grid)
        return cdf_

    def pdf(self, x):
        pdf_ = self.k_grid_product * _eval_bernstein_2d(x, self.prob_grid)
        return pdf_