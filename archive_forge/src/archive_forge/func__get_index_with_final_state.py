from statsmodels.compat.pandas import is_int_index
import contextlib
import warnings
import datetime as dt
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tools.tools import pinv_extended, Bunch
from statsmodels.tools.sm_exceptions import PrecisionWarning, ValueWarning
from statsmodels.tools.numdiff import (_get_epsilon, approx_hess_cs,
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.eval_measures import aic, aicc, bic, hqic
import statsmodels.base.wrapper as wrap
import statsmodels.tsa.base.prediction as pred
from statsmodels.base.data import PandasData
import statsmodels.tsa.base.tsa_model as tsbase
from .news import NewsResults
from .simulation_smoother import SimulationSmoother
from .kalman_smoother import SmootherResults
from .kalman_filter import INVERT_UNIVARIATE, SOLVE_LU, MEMORY_CONSERVE
from .initialization import Initialization
from .tools import prepare_exog, concat, _safe_cond, get_impact_dates
def _get_index_with_final_state(self):
    if self._index_dates:
        if isinstance(self._index, pd.DatetimeIndex):
            index = pd.date_range(start=self._index[0], periods=len(self._index) + 1, freq=self._index.freq)
        elif isinstance(self._index, pd.PeriodIndex):
            index = pd.period_range(start=self._index[0], periods=len(self._index) + 1, freq=self._index.freq)
        else:
            raise NotImplementedError
    elif isinstance(self._index, pd.RangeIndex):
        try:
            start = self._index.start
            stop = self._index.stop
            step = self._index.step
        except AttributeError:
            start = self._index._start
            stop = self._index._stop
            step = self._index._step
        index = pd.RangeIndex(start, stop + step, step)
    elif is_int_index(self._index):
        value = self._index[-1] + 1
        index = pd.Index(self._index.tolist() + [value])
    else:
        raise NotImplementedError
    return index