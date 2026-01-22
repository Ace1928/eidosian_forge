from __future__ import annotations
from statsmodels.compat.pandas import Appender, Substitution, call_cached_func
from collections import defaultdict
import datetime as dt
from itertools import combinations, product
import textwrap
from types import SimpleNamespace
from typing import (
from collections.abc import Hashable, Mapping, Sequence
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.base.data import PandasData
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary, summary_params
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.ar_model import (
from statsmodels.tsa.ardl import pss_critical_values
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import DeterministicProcess
from statsmodels.tsa.tsatools import lagmat
from_formula_doc = Docstring(ARDL.from_formula.__doc__)
from_formula_doc.replace_block("Summary", "Construct an UECM from a formula")
from_formula_doc.remove_parameters("lags")
from_formula_doc.remove_parameters("order")
from_formula_doc.insert_parameters("data", lags_param)
from_formula_doc.insert_parameters("lags", order_param)
def _construct_regressors(self, hold_back: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Construct and format model regressors"""
    self._maxlag = max(self._lags) if self._lags else 0
    dendog = np.full_like(self.data.endog, np.nan)
    dendog[1:] = np.diff(self.data.endog, axis=0)
    dlag = max(0, self._maxlag - 1)
    self._endog_reg, self._endog = lagmat(dendog, dlag, original='sep')
    self._deterministic_reg = self._deterministics.in_sample()
    orig_exog = self.data.orig_exog
    exog_pandas = isinstance(orig_exog, pd.DataFrame)
    lvl = np.full_like(self.data.endog, np.nan)
    lvl[1:] = self.data.endog[:-1]
    lvls = [lvl.copy()]
    for key, val in self._order.items():
        if val is not None:
            if exog_pandas:
                loc = orig_exog.columns.get_loc(key)
            else:
                loc = key
            lvl[1:] = self.data.exog[:-1, loc]
            lvls.append(lvl.copy())
    self._levels = np.column_stack(lvls)
    if exog_pandas:
        dexog = orig_exog.diff()
    else:
        dexog = np.full_like(self.data.exog, np.nan)
        dexog[1:] = np.diff(orig_exog, axis=0)
    adj_order = {}
    for key, val in self._order.items():
        val = None if val is None or val == [1] else val[:-1]
        adj_order[key] = val
    self._exog = self._format_exog(dexog, adj_order)
    self._blocks = {'deterministic': self._deterministic_reg, 'levels': self._levels, 'endog': self._endog_reg, 'exog': self._exog, 'fixed': self._fixed}
    blocks = [self._endog]
    for key, val in self._blocks.items():
        if key != 'exog':
            blocks.append(np.asarray(val))
        else:
            for subval in val.values():
                blocks.append(np.asarray(subval))
    y = blocks[0]
    reg = np.column_stack(blocks[1:])
    exog_maxlag = 0
    for val in self._order.values():
        exog_maxlag = max(exog_maxlag, max(val) if val is not None else 0)
    self._maxlag = max(self._maxlag, exog_maxlag)
    self._maxlag = max(self._maxlag, 1)
    if hold_back is None:
        self._hold_back = int(self._maxlag)
    if self._hold_back < self._maxlag:
        raise ValueError('hold_back must be >= the maximum lag of the endog and exog variables')
    reg = reg[self._hold_back:]
    if reg.shape[1] > reg.shape[0]:
        raise ValueError(f'The number of regressors ({reg.shape[1]}) including deterministics, lags of the endog, lags of the exogenous, and fixed regressors is larger than the sample available for estimation ({reg.shape[0]}).')
    return (np.squeeze(y)[self._hold_back:], reg)