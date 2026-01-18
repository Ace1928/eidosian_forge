import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import (assert_, assert_allclose, assert_equal,
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
def test_read_write_sio():
    eg_sio1 = BytesIO()
    with make_simple(eg_sio1, 'w'):
        str_val = eg_sio1.getvalue()
    eg_sio2 = BytesIO(str_val)
    with netcdf_file(eg_sio2) as f2:
        check_simple(f2)
    eg_sio3 = BytesIO(str_val)
    assert_raises(ValueError, netcdf_file, eg_sio3, 'r', True)
    eg_sio_64 = BytesIO()
    with make_simple(eg_sio_64, 'w', version=2) as f_64:
        str_val = eg_sio_64.getvalue()
    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64) as f_64:
        check_simple(f_64)
        assert_equal(f_64.version_byte, 2)
    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64, version=2) as f_64:
        check_simple(f_64)
        assert_equal(f_64.version_byte, 2)