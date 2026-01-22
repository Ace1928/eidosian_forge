import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _TEGMMGeneric1(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome, exclude_tmoms=False, **kwargs):
        super().__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome
        self.exclude_tmoms = exclude_tmoms
        self.__dict__.update(kwargs)
        if self.data.xnames is None:
            self.data.xnames = []
        if exclude_tmoms:
            self.k_select = 0
        else:
            self.k_select = len(res_select.model.data.param_names)
        if exclude_tmoms:
            self.prob = self.results_select.predict()
        else:
            self.prob = None

    def momcond(self, params):
        k_outcome = len(params) - self.k_select
        tm = params[:k_outcome]
        p_tm = params[k_outcome:]
        tind = self.results_select.model.endog
        if self.exclude_tmoms:
            prob = self.prob
        else:
            prob = self.results_select.model.predict(p_tm)
        moms_list = []
        mom_o = self.mom_outcome(tm, self.endog, tind, prob, weighted=True)
        moms_list.append(mom_o)
        if not self.exclude_tmoms:
            mom_t = self.results_select.model.score_obs(p_tm)
            moms_list.append(mom_t)
        moms = np.column_stack(moms_list)
        return moms