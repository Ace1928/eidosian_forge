from __future__ import annotations
from statsmodels.compat.pandas import (
from collections.abc import Iterable
import datetime
import datetime as dt
from types import SimpleNamespace
from typing import Any, Literal, cast
from collections.abc import Sequence
import warnings
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
import statsmodels.base.wrapper as wrap
from statsmodels.iolib.summary import Summary
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import eval_measures
from statsmodels.tools.decorators import cache_readonly, cache_writable
from statsmodels.tools.docstring import Docstring, remove_parameters
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.typing import (
from statsmodels.tools.validation import (
from statsmodels.tsa.arima_process import arma2ma
from statsmodels.tsa.base import tsa_model
from statsmodels.tsa.base.prediction import PredictionResults
from statsmodels.tsa.deterministic import (
from statsmodels.tsa.tsatools import freq_to_period, lagmat
import warnings
def _prepare_prediction(self, params: ArrayLike, exog: ArrayLike2D, exog_oos: ArrayLike2D, start: int | str | datetime.datetime | pd.Timestamp | None, end: int | str | datetime.datetime | pd.Timestamp | None) -> tuple[np.ndarray, np.ndarray | pd.DataFrame | None, np.ndarray | pd.DataFrame | None, int, int, int]:
    params = array_like(params, 'params')
    assert isinstance(params, np.ndarray)
    if isinstance(exog, pd.DataFrame):
        _exog = exog
    else:
        _exog = array_like(exog, 'exog', ndim=2, optional=True)
    if isinstance(exog_oos, pd.DataFrame):
        _exog_oos = exog_oos
    else:
        _exog_oos = array_like(exog_oos, 'exog_oos', ndim=2, optional=True)
    start = 0 if start is None else start
    end = self._index[-1] if end is None else end
    start, end, num_oos, _ = self._get_prediction_index(start, end)
    return (params, _exog, _exog_oos, start, end, num_oos)