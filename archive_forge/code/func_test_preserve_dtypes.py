import numpy as np
import pytest
from pandas.compat import IS64
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('op', ['sum', 'min', 'max', 'prod'])
def test_preserve_dtypes(op):
    df = pd.DataFrame({'A': ['a', 'b', 'b'], 'B': [1, None, 3], 'C': pd.array([0.1, None, 3.0], dtype='Float64')})
    result = getattr(df.C, op)()
    assert isinstance(result, np.float64)
    result = getattr(df.groupby('A'), op)()
    expected = pd.DataFrame({'B': np.array([1.0, 3.0]), 'C': pd.array([0.1, 3], dtype='Float64')}, index=pd.Index(['a', 'b'], name='A'))
    tm.assert_frame_equal(result, expected)