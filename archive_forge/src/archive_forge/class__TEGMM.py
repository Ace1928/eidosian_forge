import numpy as np
from statsmodels.compat.pandas import Substitution
from scipy.linalg import block_diag
from statsmodels.regression.linear_model import WLS
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.contrast import ContrastResults
from statsmodels.tools.docstring import indent
class _TEGMM(GMM):
    """GMM class to get cov_params for treatment effects

    This combines moment conditions for the selection/treatment model and the
    outcome model to get the standard errors for the treatment effect that
    takes the first step estimation of the treatment model into account.

    this also matches standard errors of ATE and POM in Stata

    """

    def __init__(self, endog, res_select, mom_outcome):
        super().__init__(endog, None, None)
        self.results_select = res_select
        self.mom_outcome = mom_outcome
        if self.data.xnames is None:
            self.data.xnames = []

    def momcond(self, params):
        tm = params[:2]
        p_tm = params[2:]
        tind = self.results_select.model.endog
        prob = self.results_select.model.predict(p_tm)
        momt = self.mom_outcome(tm, self.endog, tind, prob)
        moms = np.column_stack((momt, self.results_select.model.score_obs(p_tm)))
        return moms