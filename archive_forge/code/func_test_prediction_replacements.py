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
def test_prediction_replacements(data, fixed):
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    direct = res.predict()
    alt = res.predict(exog=data.x)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)
    res = ARDL(data.y, 4, data.x, [1, 3], fixed=fixed).fit()
    direct = res.predict()
    alt = res.predict(fixed=fixed)
    assert_allclose(direct, alt)
    assert_index_equal(direct.index, alt.index)