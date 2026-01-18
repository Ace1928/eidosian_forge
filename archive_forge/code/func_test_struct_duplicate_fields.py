import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_struct_duplicate_fields():
    ty = pa.struct([pa.field('x', pa.int16()), pa.field('y', pa.float32()), pa.field('x', pa.int64())])
    s = pa.scalar([('x', 1), ('y', 2.0), ('x', 3)], type=ty)
    assert list(s) == list(s.keys()) == ['x', 'y', 'x']
    assert len(s) == 3
    assert s == s
    assert list(s.items()) == [('x', pa.scalar(1, pa.int16())), ('y', pa.scalar(2.0, pa.float32())), ('x', pa.scalar(3, pa.int64()))]
    assert 'x' in s
    assert 'y' in s
    assert 'z' not in s
    assert 0 not in s
    with pytest.raises(KeyError):
        s['x']
    assert isinstance(s['y'], pa.FloatScalar)
    assert s['y'].as_py() == 2.0
    assert isinstance(s[0], pa.Int16Scalar)
    assert s[0].as_py() == 1
    assert isinstance(s[1], pa.FloatScalar)
    assert s[1].as_py() == 2.0
    assert isinstance(s[2], pa.Int64Scalar)
    assert s[2].as_py() == 3
    assert 'pyarrow.StructScalar' in repr(s)
    with pytest.raises(ValueError, match='duplicate field names'):
        s.as_py()