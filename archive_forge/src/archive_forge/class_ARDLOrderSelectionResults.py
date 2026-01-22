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
class ARDLOrderSelectionResults(AROrderSelectionResults):
    """
    Results from an ARDL order selection

    Contains the information criteria for all fitted model orders.
    """

    def __init__(self, model, ics, trend, seasonal, period):
        _ics = (((0,), (0, 0, 0)),)
        super().__init__(model, _ics, trend, seasonal, period)

        def _to_dict(d):
            return (d[0], dict(d[1:]))
        self._aic = pd.Series({v[0]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._aic.index.name = self._aic.name = 'AIC'
        self._aic = self._aic.sort_index()
        self._bic = pd.Series({v[1]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._bic.index.name = self._bic.name = 'BIC'
        self._bic = self._bic.sort_index()
        self._hqic = pd.Series({v[2]: _to_dict(k) for k, v in ics.items()}, dtype=object)
        self._hqic.index.name = self._hqic.name = 'HQIC'
        self._hqic = self._hqic.sort_index()

    @property
    def dl_lags(self) -> dict[Hashable, list[int]]:
        """The lags of exogenous variables in the selected model"""
        return self._model.dl_lags