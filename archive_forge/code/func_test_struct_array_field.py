from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_struct_array_field():
    ty = pa.struct([pa.field('x', pa.int16()), pa.field('y', pa.float32())])
    a = pa.array([(1, 2.5), (3, 4.5), (5, 6.5)], type=ty)
    x0 = a.field(0)
    y0 = a.field(1)
    x1 = a.field(-2)
    y1 = a.field(-1)
    x2 = a.field('x')
    y2 = a.field('y')
    assert isinstance(x0, pa.lib.Int16Array)
    assert isinstance(y1, pa.lib.FloatArray)
    assert x0.equals(pa.array([1, 3, 5], type=pa.int16()))
    assert y0.equals(pa.array([2.5, 4.5, 6.5], type=pa.float32()))
    assert x0.equals(x1)
    assert x0.equals(x2)
    assert y0.equals(y1)
    assert y0.equals(y2)
    for invalid_index in [None, pa.int16()]:
        with pytest.raises(TypeError):
            a.field(invalid_index)
    for invalid_index in [3, -3]:
        with pytest.raises(IndexError):
            a.field(invalid_index)
    for invalid_name in ['z', '']:
        with pytest.raises(KeyError):
            a.field(invalid_name)