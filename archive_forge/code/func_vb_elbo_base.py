from a common formula are constrained to have the same standard
import numpy as np
from scipy.optimize import minimize
from scipy import sparse
import statsmodels.base.model as base
from statsmodels.iolib import summary2
from statsmodels.genmod import families
import pandas as pd
import warnings
import patsy
def vb_elbo_base(self, h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd):
    """
        Returns the evidence lower bound (ELBO) for the model.

        This function calculates the family-specific ELBO function
        based on information provided from a subclass.

        Parameters
        ----------
        h : function mapping 1d vector to 1d vector
            The contribution of the model to the ELBO function can be
            expressed as y_i*lp_i + Eh_i(z), where y_i and lp_i are
            the response and linear predictor for observation i, and z
            is a standard normal random variable.  This formulation
            can be achieved for any GLM with a canonical link
            function.
        """
    iv = 0
    for w in glw:
        z = self.rng * w[1]
        iv += w[0] * h(z) * np.exp(-z ** 2 / 2)
    iv /= np.sqrt(2 * np.pi)
    iv *= self.rng
    iv += self.endog * tm
    iv = iv.sum()
    iv += self._elbo_common(fep_mean, fep_sd, vcp_mean, vcp_sd, vc_mean, vc_sd)
    r = iv + np.sum(np.log(fep_sd)) + np.sum(np.log(vcp_sd)) + np.sum(np.log(vc_sd))
    return r