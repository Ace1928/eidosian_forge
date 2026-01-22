import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class TreatmentEffect:
    """
    Estimate average treatment effect under conditional independence

    .. versionadded:: 0.14.0

    This class estimates treatment effect and potential outcome using 5
    different methods, ipw, ra, aipw, aipw-wls, ipw-ra.
    Standard errors and inference are based on the joint GMM representation of
    selection or treatment model, outcome model and effect functions.

    Parameters
    ----------
    model : instance of a model class
        The model class should contain endog and exog for the outcome model.
    treatment : ndarray
        indicator array for observations with treatment (1) or without (0)
    results_select : results instance
        The results instance for the treatment or selection model.
    _cov_type : "HC0"
        Internal keyword. The keyword oes not affect GMMResults which always
        corresponds to HC0 standard errors.
    kwds : keyword arguments
        currently not used

    Notes
    -----
    The outcome model is currently limited to a linear model based on OLS.
    Other outcome models, like Logit and Poisson, will become available in
    future.

    See `Treatment Effect notebook
    <../examples/notebooks/generated/treatment_effect.html>`__
    for an overview.

    """

    def __init__(self, model, treatment, results_select=None, _cov_type='HC0', **kwds):
        self.__dict__.update(kwds)
        self.treatment = np.asarray(treatment)
        self.treat_mask = treat_mask = treatment == 1
        if results_select is not None:
            self.results_select = results_select
            self.prob_select = results_select.predict()
        self.model_pool = model
        endog = model.endog
        exog = model.exog
        self.nobs = endog.shape[0]
        self._cov_type = _cov_type
        mod0 = model.__class__(endog[~treat_mask], exog[~treat_mask])
        self.results0 = mod0.fit(cov_type=_cov_type)
        mod1 = model.__class__(endog[treat_mask], exog[treat_mask])
        self.results1 = mod1.fit(cov_type=_cov_type)
        self.exog_grouped = np.concatenate((mod0.exog, mod1.exog), axis=0)
        self.endog_grouped = np.concatenate((mod0.endog, mod1.endog), axis=0)

    @classmethod
    def from_data(cls, endog, exog, treatment, model='ols', **kwds):
        """create models from data

        not yet implemented

        """
        raise NotImplementedError

    def ipw(self, return_results=True, effect_group='all', disp=False):
        """Inverse Probability Weighted treatment effect estimation.

        Parameters
        ----------
        return_results : bool
            If True, then a results instance is returned.
            If False, just ATE, POM0 and POM1 are returned.
        effect_group : {"all", 0, 1}
            ``effectgroup`` determines for which population the effects are
            estimated.
            If effect_group is "all", then sample average treatment effect and
            potential outcomes are returned.
            If effect_group is 1 or "treated", then effects on treated are
            returned.
            If effect_group is 0, "treated" or "control", then effects on
            untreated, i.e. control group, are returned.
        disp : bool
            Indicates whether the scipy optimizer should display the
            optimization results

        Returns
        -------
        TreatmentEffectsResults instance or tuple (ATE, POM0, POM1)

        See Also
        --------
        TreatmentEffectsResults
        """
        endog = self.model_pool.endog
        tind = self.treatment
        prob = self.prob_select
        if effect_group == 'all':
            probt = None
        elif effect_group in [1, 'treated']:
            probt = prob
            effect_group = 1
        elif effect_group in [0, 'untreated', 'control']:
            probt = 1 - prob
            effect_group = 0
        elif isinstance(effect_group, np.ndarray):
            probt = effect_group
            effect_group = 'user'
        else:
            raise ValueError('incorrect option for effect_group')
        res_ipw = ate_ipw(endog, tind, prob, weighted=True, probt=probt)
        if not return_results:
            return res_ipw
        gmm = _IPWGMM(endog, self.results_select, None, teff=self, effect_group=effect_group)
        start_params = np.concatenate((res_ipw[:2], self.results_select.params))
        res_gmm = gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
        res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
        return res

    @Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
    def ra(self, return_results=True, effect_group='all', disp=False):
        """
        Regression Adjustment treatment effect estimation.
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults
        """
        tind = np.zeros(len(self.treatment))
        tind[-self.treatment.sum():] = 1
        if effect_group == 'all':
            probt = None
        elif effect_group in [1, 'treated']:
            probt = tind
            effect_group = 1
        elif effect_group in [0, 'untreated', 'control']:
            probt = 1 - tind
            effect_group = 0
        elif isinstance(effect_group, np.ndarray):
            probt = effect_group
            effect_group = 'user'
        else:
            raise ValueError('incorrect option for effect_group')
        exog = self.exog_grouped
        if probt is not None:
            cw = probt / probt.mean()
        else:
            cw = 1
        pom0 = (self.results0.predict(exog) * cw).mean()
        pom1 = (self.results1.predict(exog) * cw).mean()
        if not return_results:
            return (pom1 - pom0, pom0, pom1)
        endog = self.model_pool.endog
        mod_gmm = _RAGMM(endog, self.results_select, None, teff=self, probt=probt)
        start_params = np.concatenate(([pom1 - pom0, pom0], self.results0.params, self.results1.params))
        res_gmm = mod_gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
        res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
        return res

    @Substitution(params_returns=indent(doc_params_returns2, ' ' * 8))
    def aipw(self, return_results=True, disp=False):
        """
        ATE and POM from double robust augmented inverse probability weighting
        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
        nobs = self.nobs
        prob = self.prob_select
        tind = self.treatment
        exog = self.model_pool.exog
        correct0 = (self.results0.resid / (1 - prob[tind == 0])).sum() / nobs
        correct1 = (self.results1.resid / prob[tind == 1]).sum() / nobs
        tmean0 = self.results0.predict(exog).mean() + correct0
        tmean1 = self.results1.predict(exog).mean() + correct1
        ate = tmean1 - tmean0
        if not return_results:
            return (ate, tmean0, tmean1)
        endog = self.model_pool.endog
        p2_aipw = np.asarray([ate, tmean0])
        mag_aipw1 = _AIPWGMM(endog, self.results_select, None, teff=self)
        start_params = np.concatenate((p2_aipw, self.results0.params, self.results1.params, self.results_select.params))
        res_gmm = mag_aipw1.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 5000, 'disp': disp}, maxiter=1)
        res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group='all')
        return res

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

    @Substitution(params_returns=indent(doc_params_returns, ' ' * 8))
    def ipw_ra(self, return_results=True, effect_group='all', disp=False):
        """
        ATE and POM from inverse probability weighted regression adjustment.

        
%(params_returns)s
        See Also
        --------
        TreatmentEffectsResults

        """
        treat_mask = self.treat_mask
        endog = self.model_pool.endog
        exog = self.model_pool.exog
        prob = self.prob_select
        prob0 = prob[~treat_mask]
        prob1 = prob[treat_mask]
        if effect_group == 'all':
            w0 = 1 / (1 - prob0)
            w1 = 1 / prob1
            exogt = exog
        elif effect_group in [1, 'treated']:
            w0 = prob0 / (1 - prob0)
            w1 = prob1 / prob1
            exogt = exog[treat_mask]
            effect_group = 1
        elif effect_group in [0, 'untreated', 'control']:
            w0 = (1 - prob0) / (1 - prob0)
            w1 = (1 - prob1) / prob1
            exogt = exog[~treat_mask]
            effect_group = 0
        else:
            raise ValueError('incorrect option for effect_group')
        mod0 = WLS(endog[~treat_mask], exog[~treat_mask], weights=w0)
        result0 = mod0.fit(cov_type='HC1')
        mean0_ipwra = result0.predict(exogt).mean()
        mod1 = WLS(endog[treat_mask], exog[treat_mask], weights=w1)
        result1 = mod1.fit(cov_type='HC1')
        mean1_ipwra = result1.predict(exogt).mean()
        if not return_results:
            return (mean1_ipwra - mean0_ipwra, mean0_ipwra, mean1_ipwra)
        mod_gmm = _IPWRAGMM(endog, self.results_select, None, teff=self, effect_group=effect_group)
        start_params = np.concatenate(([mean1_ipwra - mean0_ipwra, mean0_ipwra], result0.params, result1.params, np.asarray(self.results_select.params)))
        res_gmm = mod_gmm.fit(start_params=start_params, inv_weights=np.eye(len(start_params)), optim_method='nm', optim_args={'maxiter': 2000, 'disp': disp}, maxiter=1)
        res = TreatmentEffectResults(self, res_gmm, 'IPW', start_params=start_params, effect_group=effect_group)
        return res