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
class GLMResultsWrapper(lm.RegressionResultsWrapper):
    _attrs = {'resid_anscombe': 'rows', 'resid_deviance': 'rows', 'resid_pearson': 'rows', 'resid_response': 'rows', 'resid_working': 'rows'}
    _wrap_attrs = wrap.union_dicts(lm.RegressionResultsWrapper._wrap_attrs, _attrs)