import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _RAGMM(_TEGMMGeneric1):
    """GMM for regression adjustment treatment effect and potential outcome

    uses unweighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff
        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]
        k = ra.results0.model.exog.shape[1]
        pm = params[0]
        p0 = params[1:k + 1]
        p1 = params[-k:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        exog = ra.exog_grouped
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0)
        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1)
        momout = block_diag(mom0, mom1)
        mm = fitted1 - fitted0 - pm
        mpom = fitted0 - ppom
        mm = np.column_stack((mm, mpom))
        if self.probt is not None:
            mm *= (self.probt / self.probt.mean())[:, None]
        moms = np.column_stack((mm, momout))
        return moms