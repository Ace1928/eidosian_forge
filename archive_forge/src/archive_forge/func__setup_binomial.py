from statsmodels.compat.pandas import Appender
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
from statsmodels.base import _prediction_inference as pred
from statsmodels.base._prediction_inference import PredictionResultsMean
import statsmodels.base._parameter_inference as pinfer
from statsmodels.graphics._regressionplots_doc import (
import statsmodels.regression._tools as reg_tools
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (
from statsmodels.tools.docstring import Docstring
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.validation import float_like
from . import families
def _setup_binomial(self):
    self.n_trials = np.ones(self.endog.shape[0])
    if isinstance(self.family, families.Binomial):
        tmp = self.family.initialize(self.endog, self.freq_weights)
        self.endog = tmp[0]
        self.n_trials = tmp[1]
        self._init_keys.append('n_trials')