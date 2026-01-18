from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
@pytest.mark.parametrize(('data', 'dtype', 'expected_dtype'), [([1, 2, None], 'Int64', 'int64'), ([1, 2, None], 'Int64[pyarrow]', 'int64'), ([1, 2, None], 'Int8', 'int8'), ([1, 2, None], 'Int8[pyarrow]', 'int8'), ([1, 2, None], 'UInt64', 'uint64'), ([1, 2, None], 'UInt64[pyarrow]', 'uint64'), ([1.0, 2.25, None], 'Float32', 'float32'), ([1.0, 2.25, None], 'Float32[pyarrow]', 'float32'), ([True, False, None], 'boolean', 'bool'), ([True, False, None], 'boolean[pyarrow]', 'bool'), (['much ado', 'about', None], 'string[pyarrow_numpy]', 'large_string'), (['much ado', 'about', None], 'string[pyarrow]', 'large_string'), ([datetime(2020, 1, 1), datetime(2020, 1, 2), None], 'timestamp[ns][pyarrow]', 'timestamp[ns]'), ([datetime(2020, 1, 1), datetime(2020, 1, 2), None], 'timestamp[us][pyarrow]', 'timestamp[us]'), ([datetime(2020, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 2, tzinfo=timezone.utc), None], 'timestamp[us, Asia/Kathmandu][pyarrow]', 'timestamp[us, tz=Asia/Kathmandu]')])
def test_pandas_nullable_with_missing_values(data: list, dtype: str, expected_dtype: str) -> None:
    pa = pytest.importorskip('pyarrow', '11.0.0')
    import pyarrow.interchange as pai
    if expected_dtype == 'timestamp[us, tz=Asia/Kathmandu]':
        expected_dtype = pa.timestamp('us', 'Asia/Kathmandu')
    df = pd.DataFrame({'a': data}, dtype=dtype)
    result = pai.from_dataframe(df.__dataframe__())['a']
    assert result.type == expected_dtype
    assert result[0].as_py() == data[0]
    assert result[1].as_py() == data[1]
    assert result[2].as_py() is None