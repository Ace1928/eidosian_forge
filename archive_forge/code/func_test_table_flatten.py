from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_table_flatten():
    ty1 = pa.struct([pa.field('x', pa.int16()), pa.field('y', pa.float32())])
    ty2 = pa.struct([pa.field('nest', ty1)])
    a = pa.array([(1, 2.5), (3, 4.5)], type=ty1)
    b = pa.array([((11, 12.5),), ((13, 14.5),)], type=ty2)
    c = pa.array([False, True], type=pa.bool_())
    table = pa.Table.from_arrays([a, b, c], names=['a', 'b', 'c'])
    t2 = table.flatten()
    t2.validate()
    expected = pa.Table.from_arrays([pa.array([1, 3], type=pa.int16()), pa.array([2.5, 4.5], type=pa.float32()), pa.array([(11, 12.5), (13, 14.5)], type=ty1), c], names=['a.x', 'a.y', 'b.nest', 'c'])
    assert t2.equals(expected)