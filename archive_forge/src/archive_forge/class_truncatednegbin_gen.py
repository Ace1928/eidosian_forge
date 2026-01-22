import numpy as np
from scipy.stats import rv_discrete, poisson, nbinom
from scipy.special import gammaln
from scipy._lib._util import _lazywhere
from statsmodels.base.model import GenericLikelihoodModel
class truncatednegbin_gen(rv_discrete):
    """Truncated Generalized Negative Binomial (NB-P) discrete random variable
    """

    def _argcheck(self, mu, alpha, p, truncation):
        return (mu >= 0) & (truncation >= -1)

    def _get_support(self, mu, alpha, p, truncation):
        return (truncation + 1, self.b)

    def _logpmf(self, x, mu, alpha, p, truncation):
        size, prob = self.convert_params(mu, alpha, p)
        pmf = 0
        for i in range(int(np.max(truncation)) + 1):
            pmf += nbinom.pmf(i, size, prob)
        log_1_m_pmf = np.full_like(pmf, -np.inf)
        loc = pmf > 1
        log_1_m_pmf[loc] = np.nan
        loc = pmf < 1
        log_1_m_pmf[loc] = np.log(1 - pmf[loc])
        logpmf_ = nbinom.logpmf(x, size, prob) - log_1_m_pmf
        return logpmf_

    def _pmf(self, x, mu, alpha, p, truncation):
        return np.exp(self._logpmf(x, mu, alpha, p, truncation))

    def convert_params(self, mu, alpha, p):
        size = 1.0 / alpha * mu ** (2 - p)
        prob = size / (size + mu)
        return (size, prob)