import numpy as np
from statsmodels.base import model
import statsmodels.base.model as base
from statsmodels.tools.decorators import cache_readonly
from statsmodels.compat.pandas import Appender
class rv_discrete_float:
    """
    A class representing a collection of discrete distributions.

    Parameters
    ----------
    xk : 2d array_like
        The support points, should be non-decreasing within each
        row.
    pk : 2d array_like
        The probabilities, should sum to one within each row.

    Notes
    -----
    Each row of `xk`, and the corresponding row of `pk` describe a
    discrete distribution.

    `xk` and `pk` should both be two-dimensional ndarrays.  Each row
    of `pk` should sum to 1.

    This class is used as a substitute for scipy.distributions.
    rv_discrete, since that class does not allow non-integer support
    points, or vectorized operations.

    Only a limited number of methods are implemented here compared to
    the other scipy distribution classes.
    """

    def __init__(self, xk, pk):
        self.xk = xk
        self.pk = pk
        self.cpk = np.cumsum(self.pk, axis=1)

    def rvs(self, n=None):
        """
        Returns a random sample from the discrete distribution.

        A vector is returned containing a single draw from each row of
        `xk`, using the probabilities of the corresponding row of `pk`

        Parameters
        ----------
        n : not used
            Present for signature compatibility
        """
        n = self.xk.shape[0]
        u = np.random.uniform(size=n)
        ix = (self.cpk < u[:, None]).sum(1)
        ii = np.arange(n, dtype=np.int32)
        return self.xk[ii, ix]

    def mean(self):
        """
        Returns a vector containing the mean values of the discrete
        distributions.

        A vector is returned containing the mean value of each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """
        return (self.xk * self.pk).sum(1)

    def var(self):
        """
        Returns a vector containing the variances of the discrete
        distributions.

        A vector is returned containing the variance for each row of
        `xk`, using the probabilities in the corresponding row of
        `pk`.
        """
        mn = self.mean()
        xkc = self.xk - mn[:, None]
        return (self.pk * (self.xk - xkc) ** 2).sum(1)

    def std(self):
        """
        Returns a vector containing the standard deviations of the
        discrete distributions.

        A vector is returned containing the standard deviation for
        each row of `xk`, using the probabilities in the corresponding
        row of `pk`.
        """
        return np.sqrt(self.var())