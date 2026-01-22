import numpy as np
import numpy.lib.recfunctions as recf
from scipy import optimize
class TryCLogit:
    """
    Conditional Logit, data handling test

    Parameters
    ----------

    endog : array (nobs,nchoices)
        dummy encoding of realized choices
    exog_bychoices : list of arrays
        explanatory variables, one array of exog for each choice. Variables
        with common coefficients have to be first in each array
    ncommon : int
        number of explanatory variables with common coefficients

    Notes
    -----

    Utility for choice j is given by

        $V_j = X_j * beta + Z * gamma_j$

    where X_j contains generic variables (terminology Hess) that have the same
    coefficient across choices, and Z are variables, like individual-specific
    variables that have different coefficients across variables.

    If there are choice specific constants, then they should be contained in Z.
    For identification, the constant of one choice should be dropped.


    """

    def __init__(self, endog, exog_bychoices, ncommon):
        self.endog = endog
        self.exog_bychoices = exog_bychoices
        self.ncommon = ncommon
        self.nobs, self.nchoices = endog.shape
        self.nchoices = len(exog_bychoices)
        betaind = [exog_bychoices[ii].shape[1] - ncommon for ii in range(4)]
        zi = np.r_[[ncommon], ncommon + np.array(betaind).cumsum()]
        beta_indices = [np.r_[np.array([0, 1]), z[zi[ii]:zi[ii + 1]]] for ii in range(len(zi) - 1)]
        self.beta_indices = beta_indices
        beta = np.arange(7)
        betaidx_bychoices = [beta[idx] for idx in beta_indices]

    def xbetas(self, params):
        """these are the V_i
        """
        res = np.empty((self.nobs, self.nchoices))
        for choiceind in range(self.nchoices):
            res[:, choiceind] = np.dot(self.exog_bychoices[choiceind], params[self.beta_indices[choiceind]])
        return res

    def loglike(self, params):
        xb = self.xbetas(params)
        expxb = np.exp(xb)
        sumexpxb = expxb.sum(1)
        probs = expxb / expxb.sum(1)[:, None]
        loglike = (self.endog * np.log(probs)).sum(1)
        return -loglike.sum()

    def fit(self, start_params=None):
        if start_params is None:
            start_params = np.zeros(6)
        return optimize.fmin(self.loglike, start_params, maxfun=10000)