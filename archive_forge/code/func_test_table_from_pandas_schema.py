from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_table_from_pandas_schema():
    import pandas as pd
    df = pd.DataFrame(OrderedDict([('strs', ['', 'foo', 'bar']), ('floats', [4.5, 5, None])]))
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float32())])
    table = pa.Table.from_pandas(df, schema=schema)
    assert pa.types.is_float32(table.column('floats').type)
    assert table.schema.remove_metadata() == schema
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.timestamp('s'))])
    with pytest.raises((NotImplementedError, TypeError)):
        pa.Table.from_pandas(df, schema=schema)
    schema = pa.schema([('strs', pa.utf8()), ('floats', pa.float64()), ('ints', pa.int64())])
    with pytest.raises(KeyError, match='ints'):
        pa.Table.from_pandas(df, schema=schema)
    schema = pa.schema([('strs', pa.utf8())])
    table = pa.Table.from_pandas(df, schema=schema)
    assert table.num_columns == 1
    assert table.schema.remove_metadata() == schema
    assert table.column_names == ['strs']