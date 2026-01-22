import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.core.dtypes.common import (
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.integer import NUMPY_INT_TO_DTYPE
class NDArrayBacked2DTests(Dim2CompatTests):

    def test_copy_order(self, data):
        arr2d = data.repeat(2).reshape(-1, 2)
        assert arr2d._ndarray.flags['C_CONTIGUOUS']
        res = arr2d.copy()
        assert res._ndarray.flags['C_CONTIGUOUS']
        res = arr2d[::2, ::2].copy()
        assert res._ndarray.flags['C_CONTIGUOUS']
        res = arr2d.copy('F')
        assert not res._ndarray.flags['C_CONTIGUOUS']
        assert res._ndarray.flags['F_CONTIGUOUS']
        res = arr2d.copy('K')
        assert res._ndarray.flags['C_CONTIGUOUS']
        res = arr2d.T.copy('K')
        assert not res._ndarray.flags['C_CONTIGUOUS']
        assert res._ndarray.flags['F_CONTIGUOUS']
        msg = "order must be one of 'C', 'F', 'A', or 'K' \\(got 'Q'\\)"
        with pytest.raises(ValueError, match=msg):
            arr2d.copy('Q')
        arr_nc = arr2d[::2]
        assert not arr_nc._ndarray.flags['C_CONTIGUOUS']
        assert not arr_nc._ndarray.flags['F_CONTIGUOUS']
        assert arr_nc.copy()._ndarray.flags['C_CONTIGUOUS']
        assert not arr_nc.copy()._ndarray.flags['F_CONTIGUOUS']
        assert arr_nc.copy('C')._ndarray.flags['C_CONTIGUOUS']
        assert not arr_nc.copy('C')._ndarray.flags['F_CONTIGUOUS']
        assert not arr_nc.copy('F')._ndarray.flags['C_CONTIGUOUS']
        assert arr_nc.copy('F')._ndarray.flags['F_CONTIGUOUS']
        assert arr_nc.copy('K')._ndarray.flags['C_CONTIGUOUS']
        assert not arr_nc.copy('K')._ndarray.flags['F_CONTIGUOUS']