import logging
import numpy as np
from scipy.special import digamma, gammaln
from scipy import optimize
from gensim import utils, matutils
from gensim.models import ldamodel
def update_zeta(self):
    """Update the Zeta variational parameter.

        Zeta is described in the appendix and is equal to sum (exp(mean[word] + Variance[word] / 2)),
        over every time-slice. It is the value of variational parameter zeta which maximizes the lower bound.

        Returns
        -------
        list of float
            The updated zeta values for each time slice.

        """
    for j, val in enumerate(self.zeta):
        self.zeta[j] = np.sum(np.exp(self.mean[:, j + 1] + self.variance[:, j + 1] / 2))
    return self.zeta