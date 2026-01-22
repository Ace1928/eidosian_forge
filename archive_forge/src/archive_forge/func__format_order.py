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
def _format_order(exog: ArrayLike2D, order: _ARDLOrder, causal: bool) -> dict[Hashable, list[int]]:
    keys: list[Hashable]
    exog_order: dict[Hashable, int | Sequence[int] | None]
    if exog is None and order in (0, None):
        return {}
    if not isinstance(exog, pd.DataFrame):
        exog = array_like(exog, 'exog', ndim=2, maxdim=2)
        keys = list(range(exog.shape[1]))
    else:
        keys = [col for col in exog.columns]
    if order is None:
        exog_order = {k: None for k in keys}
    elif isinstance(order, Mapping):
        exog_order = order
        missing = set(keys).difference(order.keys())
        extra = set(order.keys()).difference(keys)
        if extra:
            msg = 'order dictionary contains keys for exogenous variable(s) that are not contained in exog'
            msg += ' Extra keys: '
            msg += ', '.join(list(sorted([str(v) for v in extra]))) + '.'
            raise ValueError(msg)
        if missing:
            msg = 'exog contains variables that are missing from the order dictionary.  Missing keys: '
            msg += ', '.join([str(k) for k in missing]) + '.'
            warnings.warn(msg, SpecificationWarning, stacklevel=2)
        for key in exog_order:
            _check_order(exog_order[key], causal)
    elif isinstance(order, _INT_TYPES):
        _check_order(order, causal)
        exog_order = {k: int(order) for k in keys}
    else:
        _check_order(order, causal)
        exog_order = {k: list(order) for k in keys}
    final_order: dict[Hashable, list[int]] = {}
    for key in exog_order:
        value = exog_order[key]
        if value is None:
            continue
        assert value is not None
        if isinstance(value, int):
            final_order[key] = list(range(int(causal), value + 1))
        else:
            final_order[key] = [int(lag) for lag in value]
    return final_order