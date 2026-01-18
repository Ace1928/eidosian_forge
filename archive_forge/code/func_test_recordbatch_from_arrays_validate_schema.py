from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_recordbatch_from_arrays_validate_schema():
    arr = pa.array([1, 2])
    schema = pa.schema([pa.field('f0', pa.list_(pa.utf8()))])
    with pytest.raises(NotImplementedError):
        pa.record_batch([arr], schema=schema)