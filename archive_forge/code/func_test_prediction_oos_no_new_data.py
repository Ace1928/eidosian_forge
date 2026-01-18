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
def test_prediction_oos_no_new_data(data):
    res = ARDL(data.y, 2, data.x, 3, causal=True).fit()
    val = res.forecast(1)
    assert val.shape[0] == 1
    res = ARDL(data.y, [3], data.x, [3]).fit()
    val = res.forecast(3)
    assert val.shape[0] == 3