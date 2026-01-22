import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
@Substitution(params_returns=indent(doc_params_returns2, ' ' * 8))
def aipw_wls(self, return_results=True, disp=False):
    """
        ATE and POM from double robust augmented inverse probability weighting.

        This uses weighted outcome regression, while `aipw` uses unweighted
        outcome regression.
        Option for effect on treated or on untreated is not available.
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
    nobs = self.nobs
    prob = self.prob_select
    endog = self.model_pool.endog
    exog = self.model_pool.exog
    tind = self.treatment
    treat_mask = self.treat_mask
    ww1 = tind / prob * (tind / prob - 1)
    mod1 = WLS(endog[treat_mask], exog[treat_mask], weights=ww1[treat_mask])
    result1 = mod1.fit(cov_type='HC1')
    mean1_ipw2 = result1.predict(exog).mean()
    ww0 = (1 - tind) / (1 - prob) * ((1 - tind) / (1 - prob) - 1)
    mod0 = WLS(endog[~treat_mask], exog[~treat_mask], weights=ww0[~treat_mask])
    result0 = mod0.fit(cov_type='HC1')
    mean0_ipw2 = result0.predict(exog).mean()
    self.results_ipwwls0 = result0
    self.results_ipwwls1 = result1
    correct0 = (result0.resid / (1 - prob[tind == 0])).sum() / nobs
    correct1 = (result1.resid / prob[tind == 1]).sum() / nobs
    tmean0 = mean0_ipw2 + correct0
    tmean1 = mean1_ipw2 + correct1
    ate = tmean1 - tmean0
    if not return_results:
        return (ate, tmean0, tmean1)
    p2_aipw_wls = np.asarray([ate, tmean0]).squeeze()
    mod_gmm = _AIPWWLSGMM(endog, self.results_select, None, teff=self)
    start_params = np.concatenate((p2_aipw_wls, result0.params, result1.params, self.results_select.params))
    res_gmm = mod_gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
    res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group='all')
    return res