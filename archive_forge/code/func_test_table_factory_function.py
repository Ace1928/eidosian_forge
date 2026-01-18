from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_table_factory_function():
    import pandas as pd
    d = OrderedDict([('b', ['a', 'b', 'c']), ('a', [1, 2, 3])])
    d_explicit = {'b': pa.array(['a', 'b', 'c'], type='string'), 'a': pa.array([1, 2, 3], type='int32')}
    schema = pa.schema([('a', pa.int32()), ('b', pa.string())])
    df = pd.DataFrame(d)
    table1 = pa.table(df)
    table2 = pa.Table.from_pandas(df)
    assert table1.equals(table2)
    table1 = pa.table(df, schema=schema)
    table2 = pa.Table.from_pandas(df, schema=schema)
    assert table1.equals(table2)
    table1 = pa.table(d_explicit)
    table2 = pa.Table.from_pydict(d_explicit)
    assert table1.equals(table2)
    table1 = pa.table(d, schema=schema)
    table2 = pa.Table.from_pydict(d, schema=schema)
    assert table1.equals(table2)