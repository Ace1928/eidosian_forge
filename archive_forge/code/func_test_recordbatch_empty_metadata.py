from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_empty_metadata():
    data = [pa.array(range(5)), pa.array([-10, -5, 0, 5, 10])]
    batch = pa.record_batch(data, ['c0', 'c1'])
    assert batch.schema.metadata is None