from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
@pytest.mark.parametrize('np_float', [np.float32, np.float64])
def test_pandas_to_pyarrow_with_missing(np_float):
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('__dataframe__ added to pandas in 1.5.0')
    np_array = np.array([0, np.nan, 2], dtype=np_float)
    datetime_array = [None, dt(2007, 7, 14), dt(2007, 7, 15)]
    df = pd.DataFrame({'a': np_array, 'dt': datetime_array})
    expected = pa.table({'a': pa.array(np_array, from_pandas=True), 'dt': pa.array(datetime_array, type=pa.timestamp('ns'))})
    result = pi.from_dataframe(df)
    assert result.equals(expected)