from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from pandas import MultiIndex, get_dummies
from scipy import special, stats
from scipy.special import digamma, gammaln, loggamma, polygamma
from scipy.stats import nbinom
from statsmodels.base.data import handle_data  # for mnlogit
from statsmodels.base.l1_slsqp import fit_l1_slsqp
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base._constraints import fit_constrained_wrap
import statsmodels.base._parameter_inference as pinfer
from statsmodels.base import _prediction_inference as pred
from statsmodels.distributions import genpoisson_p
import statsmodels.regression.linear_model as lm
from statsmodels.tools import data as data_tools, tools
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.numdiff import approx_fprime_cs
from statsmodels.tools.sm_exceptions import (
class NegativeBinomialResults(CountResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for NegativeBinomial 1 and 2', 'extra_attr': ''}

    @cache_readonly
    def lnalpha(self):
        """Natural log of alpha"""
        return np.log(self.params[-1])

    @cache_readonly
    def lnalpha_std_err(self):
        """Natural log of standardized error"""
        return self.bse[-1] / self.params[-1]

    @cache_readonly
    def aic(self):
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * (self.llf - (self.df_model + self.k_constant + k_extra))

    @cache_readonly
    def bic(self):
        k_extra = getattr(self.model, 'k_extra', 0)
        return -2 * self.llf + np.log(self.nobs) * (self.df_model + self.k_constant + k_extra)