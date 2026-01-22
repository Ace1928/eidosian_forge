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
def _prob_nonzero(self, mu, params):
    """Probability that count is not zero

        internal use in Censored model, will be refactored or removed
        """
    prob_nz = self.model_main._prob_nonzero(mu, params)
    return prob_nz