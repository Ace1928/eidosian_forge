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
def test_prediction_wrong_shape(data):
    x = np.asarray(data.x)
    res = ARDL(data.y, 4, x, [1, 3]).fit()
    with pytest.raises(ValueError, match='exog must have the same number'):
        res.predict(exog=np.asarray(data.x)[:, :1])
    with pytest.raises(ValueError, match='exog must have the same number of rows'):
        res.predict(exog=np.asarray(data.x)[:-2])
    res = ARDL(data.y, 4, data.x, [1, 3]).fit()
    with pytest.raises(ValueError, match='exog must have the same columns'):
        res.predict(exog=data.x.iloc[:, :1])
    with pytest.raises(ValueError, match='exog must have the same number of rows'):
        res.predict(exog=data.x.iloc[:-2])