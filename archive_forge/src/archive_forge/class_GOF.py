from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
class GOF:
    """One Sample Goodness of Fit tests

    includes Kolmogorov-Smirnov D, D+, D-, Kuiper V, Cramer-von Mises W^2, U^2 and
    Anderson-Darling A, A^2. The p-values for all tests except for A^2 are based on
    the approximatiom given in Stephens 1970. A^2 has currently no p-values. For
    the Kolmogorov-Smirnov test the tests as given in scipy.stats are also available
    as options.




    design: I might want to retest with different distributions, to calculate
    data summary statistics only once, or add separate class that holds
    summary statistics and data (sounds good).




    """

    def __init__(self, rvs, cdf, args=(), N=20):
        if isinstance(rvs, str):
            if not cdf or cdf == rvs:
                cdf = getattr(distributions, rvs).cdf
                rvs = getattr(distributions, rvs).rvs
            else:
                raise AttributeError('if rvs is string, cdf has to be the same distribution')
        if isinstance(cdf, str):
            cdf = getattr(distributions, cdf).cdf
        if callable(rvs):
            kwds = {'size': N}
            vals = np.sort(rvs(*args, **kwds))
        else:
            vals = np.sort(rvs)
            N = len(vals)
        cdfvals = cdf(vals, *args)
        self.nobs = N
        self.vals_sorted = vals
        self.cdfvals = cdfvals

    @cache_readonly
    def d_plus(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (np.arange(1.0, nobs + 1) / nobs - cdfvals).max()

    @cache_readonly
    def d_minus(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        return (cdfvals - np.arange(0.0, nobs) / nobs).max()

    @cache_readonly
    def d(self):
        return np.max([self.d_plus, self.d_minus])

    @cache_readonly
    def v(self):
        """Kuiper"""
        return self.d_plus + self.d_minus

    @cache_readonly
    def wsqu(self):
        """Cramer von Mises"""
        nobs = self.nobs
        cdfvals = self.cdfvals
        wsqu = ((cdfvals - (2.0 * np.arange(1.0, nobs + 1) - 1) / nobs / 2.0) ** 2).sum() + 1.0 / nobs / 12.0
        return wsqu

    @cache_readonly
    def usqu(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        usqu = self.wsqu - nobs * (cdfvals.mean() - 0.5) ** 2
        return usqu

    @cache_readonly
    def a(self):
        nobs = self.nobs
        cdfvals = self.cdfvals
        msum = 0
        for j in range(1, nobs):
            mj = cdfvals[j] - cdfvals[:j]
            mask = mj > 0.5
            mj[mask] = 1 - mj[mask]
            msum += mj.sum()
        a = nobs / 4.0 - 2.0 / nobs * msum
        return a

    @cache_readonly
    def asqu(self):
        """Stephens 1974, does not have p-value formula for A^2"""
        nobs = self.nobs
        cdfvals = self.cdfvals
        asqu = -((2.0 * np.arange(1.0, nobs + 1) - 1) * (np.log(cdfvals) + np.log(1 - cdfvals[::-1]))).sum() / nobs - nobs
        return asqu

    def get_test(self, testid='d', pvals='stephens70upp'):
        """

        """
        stat = getattr(self, testid)
        if pvals == 'stephens70upp':
            return (gof_pvals[pvals][testid](stat, self.nobs), stat)
        else:
            return gof_pvals[pvals][testid](stat, self.nobs)