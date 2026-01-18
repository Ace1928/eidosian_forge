from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('value', [1, 1.0, True, np.bool_(True), np.int_(1), np.float64(1)])
@pytest.mark.parametrize('asarray', [True, False])
def test_rpow_special(value, asarray):
    if asarray:
        value = np.array([value])
    result = value ** NA
    if asarray:
        result = result[0]
    elif not isinstance(value, (np.float64, np.bool_, np.int_)):
        assert isinstance(result, type(value))
    assert result == value