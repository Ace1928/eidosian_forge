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
@staticmethod
def _format_exog(exog: ArrayLike2D, order: dict[Hashable, list[int]]) -> dict[Hashable, np.ndarray]:
    """Transform exogenous variables and orders to regressors"""
    if not order:
        return {}
    max_order = 0
    for val in order.values():
        if val is not None:
            max_order = max(max(val), max_order)
    if not isinstance(exog, pd.DataFrame):
        exog = array_like(exog, 'exog', ndim=2, maxdim=2)
    exog_lags = {}
    for key in order:
        if order[key] is None:
            continue
        if isinstance(exog, np.ndarray):
            assert isinstance(key, int)
            col = exog[:, key]
        else:
            col = exog[key]
        lagged_col = lagmat(col, max_order, original='in')
        lags = order[key]
        exog_lags[key] = lagged_col[:, lags]
    return exog_lags