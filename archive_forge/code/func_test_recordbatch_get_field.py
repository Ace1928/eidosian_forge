from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_get_field():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10]), pa.array(range(5, 10))]
    batch = pa.RecordBatch.from_arrays(data, names=('a', 'b', 'c'))
    assert batch.field('a').equals(batch.schema.field('a'))
    assert batch.field(0).equals(batch.schema.field('a'))
    with pytest.raises(KeyError):
        batch.field('d')
    with pytest.raises(TypeError):
        batch.field(None)
    with pytest.raises(IndexError):
        batch.field(4)