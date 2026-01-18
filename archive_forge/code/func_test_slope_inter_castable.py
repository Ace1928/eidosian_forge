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
def test_slope_inter_castable():
    for in_dtt in FLOAT_TYPES + IUINT_TYPES:
        for out_dtt in NUMERIC_TYPES:
            for klass in (ArrayWriter, SlopeArrayWriter, SlopeInterArrayWriter):
                arr = np.zeros((5,), dtype=in_dtt)
                aw = klass(arr, out_dtt)
    arr = np.array([np.inf, np.nan, -np.inf])
    for in_dtt in FLOAT_TYPES:
        for out_dtt in IUINT_TYPES:
            in_arr = arr.astype(in_dtt)
            with pytest.raises(WriterError):
                ArrayWriter(in_arr, out_dtt)
            aw = SlopeArrayWriter(arr.astype(in_dtt), out_dtt)
            aw = SlopeInterArrayWriter(arr.astype(in_dtt), out_dtt)
    for in_dtt, out_dtt, arr, slope_only, slope_inter, neither in ((np.float32, np.float32, 1, True, True, True), (np.float64, np.float32, 1, True, True, True), (np.float32, np.complex128, 1, True, True, True), (np.uint32, np.complex128, 1, True, True, True), (np.int64, np.float32, 1, True, True, True), (np.float32, np.int16, 1, True, True, False), (np.complex128, np.float32, 1, False, False, False), (np.complex128, np.int16, 1, False, False, False), (np.uint8, np.int16, 1, True, True, True), (np.uint16, np.int16, 1, True, True, True), (np.uint16, np.int16, 2 ** 16 - 1, True, True, False), (np.uint16, np.int16, (0, 2 ** 16 - 1), True, True, False), (np.uint16, np.uint8, 1, True, True, True), (np.int16, np.uint16, 1, True, True, True), (np.int16, np.uint16, -1, True, True, False), (np.int16, np.uint16, (-1, 1), False, True, False), (np.int8, np.uint16, 1, True, True, True), (np.int8, np.uint16, -1, True, True, False), (np.int8, np.uint16, (-1, 1), False, True, False)):
        data = np.array(arr, dtype=in_dtt)
        if slope_only:
            SlopeArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                SlopeArrayWriter(data, out_dtt)
        if slope_inter:
            SlopeInterArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                SlopeInterArrayWriter(data, out_dtt)
        if neither:
            ArrayWriter(data, out_dtt)
        else:
            with pytest.raises(WriterError):
                ArrayWriter(data, out_dtt)