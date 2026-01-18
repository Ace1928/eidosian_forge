import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_zero_byte_string():
    str_io = BytesIO()
    r = _make_readerlike(str_io, boc.native_code)
    c_reader = m5u.VarReader5(r)
    tag_dt = np.dtype([('mdtype', 'u4'), ('byte_count', 'u4')])
    tag = np.zeros((1,), dtype=tag_dt)
    tag['mdtype'] = mio5p.miINT8
    tag['byte_count'] = 1
    hdr = m5u.VarHeader5()
    hdr.set_dims([1])
    _write_stream(str_io, tag.tobytes() + b'        ')
    str_io.seek(0)
    val = c_reader.read_char(hdr)
    assert_equal(val, ' ')
    tag['byte_count'] = 0
    _write_stream(str_io, tag.tobytes())
    str_io.seek(0)
    val = c_reader.read_char(hdr)
    assert_equal(val, ' ')
    str_io.seek(0)
    hdr.set_dims([4])
    val = c_reader.read_char(hdr)
    assert_array_equal(val, [' '] * 4)