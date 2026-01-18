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
def vb_elbo(self, vb_mean, vb_sd):
    """
        Returns the evidence lower bound (ELBO) for the model.
        """
    fep_mean, vcp_mean, vc_mean = self._unpack(vb_mean)
    fep_sd, vcp_sd, vc_sd = self._unpack(vb_sd)
    tm, tv = self._lp_stats(fep_mean, fep_sd, vc_mean, vc_sd)

    def h(z):
        return -np.exp(tm + np.sqrt(tv) * z)
    return self.vb_elbo_base(h, tm, fep_mean, vcp_mean, vc_mean, fep_sd, vcp_sd, vc_sd)