import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops
@pytest.mark.parametrize('dropna', [True, False])
def test_nunique_dropna(dropna):
    ser = pd.Series(['yes', 'yes', pd.NA, np.nan, None, pd.NaT])
    res = ser.nunique(dropna)
    assert res == 1 if dropna else 5