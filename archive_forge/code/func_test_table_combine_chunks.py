from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_combine_chunks():
    batch1 = pa.record_batch([pa.array([1]), pa.array(['a'])], names=['f1', 'f2'])
    batch2 = pa.record_batch([pa.array([2]), pa.array(['b'])], names=['f1', 'f2'])
    table = pa.Table.from_batches([batch1, batch2])
    combined = table.combine_chunks()
    combined.validate()
    assert combined.equals(table)
    for c in combined.columns:
        assert c.num_chunks == 1