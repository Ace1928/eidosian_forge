from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_run_end_encoded_type():
    ty = pa.run_end_encoded(pa.int64(), pa.utf8())
    assert isinstance(ty, pa.RunEndEncodedType)
    assert ty.run_end_type == pa.int64()
    assert ty.value_type == pa.utf8()
    assert ty.num_buffers == 1
    assert ty.num_fields == 2
    with pytest.raises(TypeError):
        pa.run_end_encoded(pa.int64(), None)
    with pytest.raises(TypeError):
        pa.run_end_encoded(None, pa.utf8())
    with pytest.raises(ValueError):
        pa.run_end_encoded(pa.int8(), pa.utf8())