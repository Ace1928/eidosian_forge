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
class ProbitResults(BinaryResults):
    __doc__ = _discrete_results_docs % {'one_line_description': 'A results class for Probit Model', 'extra_attr': ''}

    @cache_readonly
    def resid_generalized(self):
        """
        Generalized residuals

        Notes
        -----
        The generalized residuals for the Probit model are defined

        .. math:: y\\frac{\\phi(X\\beta)}{\\Phi(X\\beta)}-(1-y)\\frac{\\phi(X\\beta)}{1-\\Phi(X\\beta)}
        """
        model = self.model
        endog = model.endog
        XB = self.predict(which='linear')
        pdf = model.pdf(XB)
        cdf = model.cdf(XB)
        return endog * pdf / cdf - (1 - endog) * pdf / (1 - cdf)