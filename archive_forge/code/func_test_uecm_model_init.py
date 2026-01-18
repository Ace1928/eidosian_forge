from typing import NamedTuple
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_index_equal
import pytest
from statsmodels.datasets import danish_data
from statsmodels.iolib.summary import Summary
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.ardl.model import (
from statsmodels.tsa.deterministic import DeterministicProcess
def test_uecm_model_init(data: Dataset, uecm_lags, uecm_order, trend, causal, fixed, use_numpy, seasonal):
    y, x, z, uecm_order, period = _convert_to_numpy(data, fixed, uecm_order, seasonal, use_numpy)
    mod = UECM(y, uecm_lags, x, uecm_order, trend, causal=causal, fixed=z, seasonal=seasonal, period=period)
    res = mod.fit()
    check_results(res)
    res.predict()
    ardl = ARDL(y, uecm_lags, x, uecm_order, trend, causal=causal, fixed=z, seasonal=seasonal, period=period)
    uecm = UECM.from_ardl(ardl)
    uecm_res = uecm.fit()
    check_results(uecm_res)
    uecm_res.predict()