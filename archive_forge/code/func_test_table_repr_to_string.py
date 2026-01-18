from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_repr_to_string():
    schema = pa.schema([pa.field('c0', pa.int16(), metadata={'key': 'value'}), pa.field('c1', pa.int32())], metadata={b'foo': b'bar'})
    tab = pa.table([pa.array([1, 2, 3, 4], type='int16'), pa.array([10, 20, 30, 40], type='int32')], schema=schema)
    assert str(tab) == 'pyarrow.Table\nc0: int16\nc1: int32\n----\nc0: [[1,2,3,4]]\nc1: [[10,20,30,40]]'
    assert tab.to_string(show_metadata=True) == "pyarrow.Table\nc0: int16\n  -- field metadata --\n  key: 'value'\nc1: int32\n-- schema metadata --\nfoo: 'bar'"
    assert tab.to_string(preview_cols=5) == 'pyarrow.Table\nc0: int16\nc1: int32\n----\nc0: [[1,2,3,4]]\nc1: [[10,20,30,40]]'
    assert tab.to_string(preview_cols=1) == 'pyarrow.Table\nc0: int16\nc1: int32\n----\nc0: [[1,2,3,4]]\n...'