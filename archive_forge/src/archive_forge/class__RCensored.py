import warnings
import numpy as np
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.distributions.discrete import (
from statsmodels.discrete.discrete_model import (
from statsmodels.tools.numdiff import approx_hess
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from copy import deepcopy
class _RCensored(_RCensoredGeneric):
    __doc__ = '\n    Censored model for count data\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, model=Poisson, distribution=truncatedpoisson, offset=None, exposure=None, missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, missing=missing, **kwargs)
        self.model_main = model(np.zeros_like(self.endog), self.exog)
        self.model_dist = distribution
        self.k_extra = k_extra = self.model_main.k_extra
        if k_extra > 0:
            self.exog_names.extend(self.model_main.exog_names[-k_extra:])
        self.result_class = TruncatedLFGenericResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper

    def _prob_nonzero(self, mu, params):
        """Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
        prob_nz = self.model_main._prob_nonzero(mu, params)
        return prob_nz