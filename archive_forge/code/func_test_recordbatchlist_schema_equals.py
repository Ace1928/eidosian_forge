from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatchlist_schema_equals():
    a1 = np.array([1], dtype='uint32')
    a2 = np.array([4.0, 5.0], dtype='float64')
    batch1 = pa.record_batch([pa.array(a1)], ['c1'])
    batch2 = pa.record_batch([pa.array(a2)], ['c1'])
    with pytest.raises(pa.ArrowInvalid):
        pa.Table.from_batches([batch1, batch2])