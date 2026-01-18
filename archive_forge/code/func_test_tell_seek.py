import os
import zlib
from io import BytesIO
from tempfile import mkstemp
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_equal
from pytest import raises as assert_raises
from scipy.io.matlab._streams import (make_stream,
def test_tell_seek():
    with setup_test_file() as (fs, gs, cs):
        for s in (fs, gs, cs):
            st = make_stream(s)
            res = st.seek(0)
            assert_equal(res, 0)
            assert_equal(st.tell(), 0)
            res = st.seek(5)
            assert_equal(res, 0)
            assert_equal(st.tell(), 5)
            res = st.seek(2, 1)
            assert_equal(res, 0)
            assert_equal(st.tell(), 7)
            res = st.seek(-2, 2)
            assert_equal(res, 0)
            assert_equal(st.tell(), 6)