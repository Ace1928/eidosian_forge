from collections import OrderedDict
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
from scipy.stats import norm, rv_continuous, rv_discrete
from scipy.stats.distributions import rv_frozen
from statsmodels.base.covtype import descriptions
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import forg
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.tools import Bunch
from statsmodels.tools.validation import (
import statsmodels.tsa.base.tsa_model as tsbase
from statsmodels.tsa.exponential_smoothing import base
import statsmodels.tsa.exponential_smoothing._ets_smooth as smooth
from statsmodels.tsa.exponential_smoothing.initialization import (
from statsmodels.tsa.tsatools import freq_to_period
def _handle_prediction_index(self, start, dynamic, end, index):
    if start is None:
        start = 0
    start, end, out_of_sample, _ = self.model._get_prediction_index(start, end, index)
    if start > end + out_of_sample + 1:
        raise ValueError('Prediction start cannot lie outside of the sample.')
    if isinstance(dynamic, (str, dt.datetime, pd.Timestamp)):
        dynamic, _, _ = self.model._get_index_loc(dynamic)
        dynamic = dynamic - start
    elif isinstance(dynamic, bool):
        if dynamic:
            dynamic = 0
        else:
            dynamic = end + 1 - start
    if dynamic == 0:
        start_smooth = None
        end_smooth = None
        nsmooth = 0
        start_dynamic = start
    else:
        start_smooth = start
        end_smooth = min(start + dynamic - 1, end)
        nsmooth = max(end_smooth - start_smooth + 1, 0)
        start_dynamic = start + dynamic
    if start_dynamic == 0:
        anchor_dynamic = 'start'
    else:
        anchor_dynamic = start_dynamic - 1
    end_dynamic = end + out_of_sample
    ndynamic = end_dynamic - start_dynamic + 1
    return (start, end, start_smooth, end_smooth, anchor_dynamic, start_dynamic, end_dynamic, nsmooth, ndynamic, index)