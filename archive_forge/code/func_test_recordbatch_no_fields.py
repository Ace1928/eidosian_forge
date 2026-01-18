from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_no_fields():
    batch = pa.record_batch([], [])
    assert len(batch) == 0
    assert batch.num_rows == 0
    assert batch.num_columns == 0