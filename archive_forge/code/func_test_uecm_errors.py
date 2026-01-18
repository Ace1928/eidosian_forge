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
def test_uecm_errors(data):
    with pytest.raises(TypeError, match='order must be None'):
        UECM(data.y, 2, data.x, [0, 1, 2])
    with pytest.raises(TypeError, match='lags must be an'):
        UECM(data.y, [1, 2], data.x, 2)
    with pytest.raises(TypeError, match='order values must be positive'):
        UECM(data.y, 2, data.x, {'ibo': [1, 2]})
    with pytest.raises(ValueError, match='Model must contain'):
        UECM(data.y, 2, data.x, None)
    with pytest.raises(ValueError, match='All included exog'):
        UECM(data.y, 2, data.x, {'lry': 2, 'ide': 2, 'ibo': 0})
    with pytest.raises(ValueError, match='hold_back must be'):
        UECM(data.y, 3, data.x, 5, hold_back=4)
    with pytest.raises(ValueError, match='The number of'):
        UECM(data.y, 20, data.x, 4)
    ardl = ARDL(data.y, 2, data.x, {'lry': [1, 2], 'ide': 2, 'ibo': 2})
    with pytest.raises(ValueError, match='UECM can only be created from'):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, 2, data.x, {'lry': [0, 2], 'ide': 2, 'ibo': 2})
    with pytest.raises(ValueError, match='UECM can only be created from'):
        UECM.from_ardl(ardl)
    ardl = ARDL(data.y, [1, 3], data.x, 2)
    with pytest.raises(ValueError, match='UECM can only be created from'):
        UECM.from_ardl(ardl)
    res = UECM(data.y, 2, data.x, 2).fit()
    with pytest.raises(NotImplementedError):
        res.predict(end=100)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=True)
    with pytest.raises(NotImplementedError):
        res.predict(dynamic=25)