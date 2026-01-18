import itertools
from io import BytesIO
from platform import machine, python_compiler
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..arraywriters import (
from ..casting import int_abs, sctypes, shared_range, type_info
from ..testing import assert_allclose_safely, suppress_warnings
from ..volumeutils import _dt_min_max, apply_read_scaling, array_from_file
def test_to_float():
    start, stop = (0, 100)
    for in_type in NUMERIC_TYPES:
        step = 1 if in_type in IUINT_TYPES else 0.5
        info = type_info(in_type)
        mn, mx = (info['min'], info['max'])
        arr = np.arange(start, stop, step, dtype=in_type)
        arr[0] = mn
        arr[-1] = mx
        for out_type in CFLOAT_TYPES:
            out_info = type_info(out_type)
            for klass in (SlopeInterArrayWriter, SlopeArrayWriter, ArrayWriter):
                if in_type in COMPLEX_TYPES and out_type in FLOAT_TYPES:
                    with pytest.raises(WriterError):
                        klass(arr, out_type)
                    continue
                aw = klass(arr, out_type)
                assert aw.array is arr
                assert aw.out_dtype == out_type
                arr_back = round_trip(aw)
                assert_array_equal(arr.astype(out_type), arr_back)
                out_min, out_max = (out_info['min'], out_info['max'])
                assert np.all(arr_back[arr > out_max] == np.inf)
                assert np.all(arr_back[arr < out_min] == -np.inf)