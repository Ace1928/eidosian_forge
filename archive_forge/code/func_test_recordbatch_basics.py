from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_basics():
    data = [pa.array(range(5), type='int16'), pa.array([-10, -5, 0, None, 10], type='int32')]
    batch = pa.record_batch(data, ['c0', 'c1'])
    assert not batch.schema.metadata
    assert len(batch) == 5
    assert batch.num_rows == 5
    assert batch.num_columns == len(data)
    assert batch.get_total_buffer_size() == 5 * 2 + (5 * 4 + 1)
    batch.nbytes == 5 * 2 + (5 * 4 + 1)
    assert sys.getsizeof(batch) >= object.__sizeof__(batch) + batch.get_total_buffer_size()
    pydict = batch.to_pydict()
    assert pydict == OrderedDict([('c0', [0, 1, 2, 3, 4]), ('c1', [-10, -5, 0, None, 10])])
    assert isinstance(pydict, dict)
    with pytest.raises(IndexError):
        batch[2]
    schema = pa.schema([pa.field('c0', pa.int16(), metadata={'key': 'value'}), pa.field('c1', pa.int32())], metadata={b'foo': b'bar'})
    batch = pa.record_batch(data, schema=schema)
    assert batch.schema == schema
    batch = pa.record_batch(data, schema)
    assert batch.schema == schema
    assert str(batch) == 'pyarrow.RecordBatch\nc0: int16\nc1: int32\n----\nc0: [0,1,2,3,4]\nc1: [-10,-5,0,null,10]'
    assert batch.to_string(show_metadata=True) == "pyarrow.RecordBatch\nc0: int16\n  -- field metadata --\n  key: 'value'\nc1: int32\n-- schema metadata --\nfoo: 'bar'"
    wr = weakref.ref(batch)
    assert wr() is not None
    del batch
    assert wr() is None