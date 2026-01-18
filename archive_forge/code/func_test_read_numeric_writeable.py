import sys
from io import BytesIO
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_
from pytest import raises as assert_raises
import scipy.io.matlab._byteordercodes as boc
import scipy.io.matlab._streams as streams
import scipy.io.matlab._mio5_params as mio5p
import scipy.io.matlab._mio5_utils as m5u
def test_read_numeric_writeable():
    str_io = BytesIO()
    r = _make_readerlike(str_io, '<')
    c_reader = m5u.VarReader5(r)
    dt = np.dtype('<u2')
    a = _make_tag(dt, 30, mio5p.miUINT16, 0)
    a_str = a.tobytes()
    _write_stream(str_io, a_str)
    el = c_reader.read_numeric()
    assert_(el.flags.writeable is True)