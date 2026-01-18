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
def test_ticket_1720():
    io = BytesIO()
    items = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with netcdf_file(io, 'w') as f:
        f.history = 'Created for a test'
        f.createDimension('float_var', 10)
        float_var = f.createVariable('float_var', 'f', ('float_var',))
        float_var[:] = items
        float_var.units = 'metres'
        f.flush()
        contents = io.getvalue()
    io = BytesIO(contents)
    with netcdf_file(io, 'r') as f:
        assert_equal(f.history, b'Created for a test')
        float_var = f.variables['float_var']
        assert_equal(float_var.units, b'metres')
        assert_equal(float_var.shape, (10,))
        assert_allclose(float_var[:], items)