from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_function_unicode_schema():
    col_a = 'äääh'
    col_b = 'öööf'
    d = OrderedDict([(col_b, ['a', 'b', 'c']), (col_a, [1, 2, 3])])
    schema = pa.schema([(col_a, pa.int32()), (col_b, pa.string())])
    result = pa.table(d, schema=schema)
    assert result[0].chunk(0).equals(pa.array([1, 2, 3], type='int32'))
    assert result[1].chunk(0).equals(pa.array(['a', 'b', 'c'], type='string'))