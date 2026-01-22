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
class TruncatedLFGeneralizedPoisson(TruncatedLFGeneric):
    __doc__ = '\n    Truncated Generalized Poisson model for count data\n\n    .. versionadded:: 0.14.0\n\n    %(params)s\n    %(extra_params)s\n\n    Attributes\n    ----------\n    endog : array\n        A reference to the endogenous response variable\n    exog : array\n        A reference to the exogenous design.\n    truncation : int, optional\n        Truncation parameter specify truncation point out of the support\n        of the distribution. pmf(k) = 0 for k <= truncation\n    ' % {'params': base._model_params_doc, 'extra_params': 'offset : array_like\n        Offset is added to the linear prediction with coefficient equal to 1.\n    exposure : array_like\n        Log(exposure) is added to the linear prediction with coefficient\n        equal to 1.\n\n    ' + base._missing_param_doc}

    def __init__(self, endog, exog, offset=None, exposure=None, truncation=0, p=2, missing='none', **kwargs):
        super().__init__(endog, exog, offset=offset, exposure=exposure, truncation=truncation, missing=missing, **kwargs)
        self.model_main = GeneralizedPoisson(self.endog, self.exog, exposure=getattr(self, 'exposure', None), offset=getattr(self, 'offset', None), p=p)
        self.k_extra = self.model_main.k_extra
        self.exog_names.extend(self.model_main.exog_names[-self.k_extra:])
        self.model_dist = None
        self.result_class = TruncatedNegativeBinomialResults
        self.result_class_wrapper = TruncatedLFGenericResultsWrapper
        self.result_class_reg = L1TruncatedLFGenericResults
        self.result_class_reg_wrapper = L1TruncatedLFGenericResultsWrapper