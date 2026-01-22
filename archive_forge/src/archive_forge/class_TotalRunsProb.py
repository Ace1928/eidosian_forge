import numpy as np
from scipy import stats
from scipy.special import comb
import warnings
from statsmodels.tools.validation import array_like
class TotalRunsProb:
    """class for the probability distribution of total runs

    This is the exact probability distribution for the (Wald-Wolfowitz)
    runs test. The random variable is the total number of runs if the
    sample has (n0, n1) observations of groups 0 and 1.


    Notes
    -----
    Written as a class so I can store temporary calculations, but I do not
    think it matters much.

    Formulas taken from SAS manual for one-sided significance level.

    Could be converted to a full univariate distribution, subclassing
    scipy.stats.distributions.

    *Status*
    Not verified yet except for mean.



    """

    def __init__(self, n0, n1):
        self.n0 = n0
        self.n1 = n1
        self.n = n = n0 + n1
        self.comball = comb(n, n1)

    def runs_prob_even(self, r):
        n0, n1 = (self.n0, self.n1)
        tmp0 = comb(n0 - 1, r // 2 - 1)
        tmp1 = comb(n1 - 1, r // 2 - 1)
        return tmp0 * tmp1 * 2.0 / self.comball

    def runs_prob_odd(self, r):
        n0, n1 = (self.n0, self.n1)
        k = (r + 1) // 2
        tmp0 = comb(n0 - 1, k - 1)
        tmp1 = comb(n1 - 1, k - 2)
        tmp3 = comb(n0 - 1, k - 2)
        tmp4 = comb(n1 - 1, k - 1)
        return (tmp0 * tmp1 + tmp3 * tmp4) / self.comball

    def pdf(self, r):
        r = np.asarray(r)
        r_isodd = np.mod(r, 2) > 0
        r_odd = r[r_isodd]
        r_even = r[~r_isodd]
        runs_pdf = np.zeros(r.shape)
        runs_pdf[r_isodd] = self.runs_prob_odd(r_odd)
        runs_pdf[~r_isodd] = self.runs_prob_even(r_even)
        return runs_pdf

    def cdf(self, r):
        r_ = np.arange(2, r + 1)
        cdfval = self.runs_prob_even(r_[::2]).sum()
        cdfval += self.runs_prob_odd(r_[1::2]).sum()
        return cdfval