from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
class TestFromNumPyDtype:

    def test_int32(self):
        assert from_numpy((2,), 'int32') == dshape('2 * int32')
        assert from_numpy((2,), 'i4') == dshape('2 * int32')

    def test_struct(self):
        dtype = np.dtype([('x', '<i4'), ('y', '<i4')])
        result = from_numpy((2,), dtype)
        assert result == dshape('2 * {x: int32, y: int32}')

    def test_datetime(self):
        keys = ('h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as')
        for k in keys:
            assert from_numpy((2,), np.dtype('M8[%s]' % k)) == dshape('2 * datetime')

    def test_date(self):
        for d in ('D', 'M', 'Y', 'W'):
            assert from_numpy((2,), np.dtype('M8[%s]' % d)) == dshape('2 * date')

    def test_timedelta(self):
        for d in _units:
            assert from_numpy((2,), np.dtype('m8[%s]' % d)) == dshape('2 * timedelta[unit=%r]' % d)

    def test_ascii_string(self):
        assert from_numpy((2,), np.dtype('S7')) == dshape('2 * string[7, "ascii"]')

    def test_string(self):
        assert from_numpy((2,), np.dtype('U7')) == dshape('2 * string[7, "U32"]')

    def test_string_from_CType_classmethod(self):
        assert CType.from_numpy_dtype(np.dtype('S7')) == String(7, 'A')