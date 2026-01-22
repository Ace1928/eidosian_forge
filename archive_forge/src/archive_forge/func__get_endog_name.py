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
def _get_endog_name(self, yname, yname_list, all=False):
    """
        If all is False, the first variable name is dropped
        """
    model = self.model
    if yname is None:
        yname = model.endog_names
    if yname_list is None:
        ynames = model._ynames_map
        ynames = self._maybe_convert_ynames_int(ynames)
        ynames = [ynames[key] for key in range(int(model.J))]
        ynames = ['='.join([yname, name]) for name in ynames]
        if not all:
            yname_list = ynames[1:]
        else:
            yname_list = ynames
    return (yname, yname_list)