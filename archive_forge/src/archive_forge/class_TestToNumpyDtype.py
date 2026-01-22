from collections import OrderedDict
import datetime
from operator import getitem
import pickle
import numpy as np
import pytest
from datashader.datashape.coretypes import (
from datashader.datashape import (
class TestToNumpyDtype:

    def test_simple(self):
        assert to_numpy_dtype(dshape('2 * int32')) == np.int32
        assert to_numpy_dtype(dshape('2 * {x: int32, y: int32}')) == np.dtype([('x', '<i4'), ('y', '<i4')])

    def test_datetime(self):
        assert to_numpy_dtype(dshape('2 * datetime')) == np.dtype('M8[us]')

    def test_date(self):
        assert to_numpy_dtype(dshape('2 * date')) == np.dtype('M8[D]')

    def test_string(self):
        assert to_numpy_dtype(dshape('2 * string')) == np.dtype('O')

    def test_dimensions(self):
        assert to_numpy_dtype(dshape('var * int32')) == np.int32

    def test_timedelta(self):
        assert to_numpy_dtype(dshape('2 * timedelta')) == np.dtype('m8[us]')
        assert to_numpy_dtype(dshape("2 * timedelta[unit='s']")) == np.dtype('m8[s]')

    def test_decimal(self):
        assert to_numpy_dtype(dshape('decimal[18,0]')) == np.int64
        assert to_numpy_dtype(dshape('decimal[7,2]')) == np.float64
        assert to_numpy_dtype(dshape('decimal[4]')) == np.int16
        with pytest.raises(TypeError):
            to_numpy_dtype(dshape('decimal[21]'))