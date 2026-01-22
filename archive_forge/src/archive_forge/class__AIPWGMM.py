import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _AIPWGMM(_TEGMMGeneric1):
    """ GMM for aipw treatment effect and potential outcome

    uses unweighted outcome regression
    """

    def momcond(self, params):
        ra = self.teff
        treat_mask = ra.treat_mask
        res_select = ra.results_select
        ppom = params[1]
        mask = np.arange(len(params)) != 1
        params = params[mask]
        k = ra.results0.model.exog.shape[1]
        pm = params[0]
        p0 = params[1:k + 1]
        p1 = params[k + 1:2 * k + 1]
        ps = params[2 * k + 1:]
        mod0 = ra.results0.model
        mod1 = ra.results1.model
        exog = ra.exog_grouped
        endog = ra.endog_grouped
        prob_sel = np.asarray(res_select.model.predict(ps))
        prob_sel = np.clip(prob_sel, 0.01, 0.99)
        prob0 = prob_sel[~treat_mask]
        prob1 = prob_sel[treat_mask]
        prob = np.concatenate((prob0, prob1))
        fitted0 = mod0.predict(p0, exog)
        mom0 = _mom_olsex(p0, model=mod0)
        fitted1 = mod1.predict(p1, exog)
        mom1 = _mom_olsex(p1, model=mod1)
        mom_outcome = block_diag(mom0, mom1)
        tind = ra.treatment
        tind = np.concatenate((tind[~treat_mask], tind[treat_mask]))
        correct0 = (endog - fitted0) / (1 - prob) * (1 - tind)
        correct1 = (endog - fitted1) / prob * tind
        tmean0 = fitted0 + correct0
        tmean1 = fitted1 + correct1
        ate = tmean1 - tmean0
        mm = ate - pm
        mpom = tmean0 - ppom
        mm = np.column_stack((mm, mpom))
        mom_select = res_select.model.score_obs(ps)
        mom_select = np.concatenate((mom_select[~treat_mask], mom_select[treat_mask]), axis=0)
        moms = np.column_stack((mm, mom_outcome, mom_select))
        return moms