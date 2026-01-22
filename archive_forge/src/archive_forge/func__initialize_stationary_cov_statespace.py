import numpy as np
import pandas as pd
from statsmodels.base.data import PandasData
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.validation import (array_like, bool_like, float_like,
from statsmodels.tsa.exponential_smoothing import initialization as es_init
from statsmodels.tsa.statespace import initialization as ss_init
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.compat.pandas import Appender
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from .mlemodel import MLEModel, MLEResults, MLEResultsWrapper
def _initialize_stationary_cov_statespace(self):
    R = self.ssm['selection']
    Q = self.ssm['state_cov']
    self.initialization.stationary_cov = R.dot(Q).dot(R.T)