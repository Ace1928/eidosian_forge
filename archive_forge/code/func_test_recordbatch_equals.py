from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_equals():
    data1 = [pa.array(range(5), type='int16'), pa.array([-10, -5, 0, None, 10], type='int32')]
    data2 = [pa.array(['a', 'b', 'c']), pa.array([['d'], ['e'], ['f']])]
    column_names = ['c0', 'c1']
    batch = pa.record_batch(data1, column_names)
    assert batch == pa.record_batch(data1, column_names)
    assert batch.equals(pa.record_batch(data1, column_names))
    assert batch != pa.record_batch(data2, column_names)
    assert not batch.equals(pa.record_batch(data2, column_names))
    batch_meta = pa.record_batch(data1, names=column_names, metadata={'key': 'value'})
    assert batch_meta.equals(batch)
    assert not batch_meta.equals(batch, check_metadata=True)
    assert not batch.equals(None)
    assert batch != 'foo'