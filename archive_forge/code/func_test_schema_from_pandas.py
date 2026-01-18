from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_schema_from_pandas():
    import pandas as pd
    inputs = [list(range(10)), pd.Categorical(list(range(10))), ['foo', 'bar', None, 'baz', 'qux'], np.array(['2007-07-13T01:23:34.123456789', '2006-01-13T12:34:56.432539784', '2010-08-13T05:46:57.437699912'], dtype='datetime64[ns]'), pd.array([1, 2, None], dtype=pd.Int32Dtype())]
    for data in inputs:
        df = pd.DataFrame({'a': data}, index=data)
        schema = pa.Schema.from_pandas(df)
        expected = pa.Table.from_pandas(df).schema
        assert schema == expected