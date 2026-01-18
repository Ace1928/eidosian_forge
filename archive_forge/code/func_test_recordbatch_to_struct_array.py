from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_to_struct_array():
    batch = pa.RecordBatch.from_arrays([pa.array([1, None], type=pa.int32()), pa.array([None, 1.0], type=pa.float32())], ['ints', 'floats'])
    result = batch.to_struct_array()
    assert result.equals(pa.array([{'ints': 1}, {'floats': 1.0}], type=pa.struct([('ints', pa.int32()), ('floats', pa.float32())])))