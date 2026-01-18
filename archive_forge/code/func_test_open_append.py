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
def test_open_append():
    with in_tempdir():
        filename = 'append_dat.nc'
        f = netcdf_file(filename, 'w')
        f._attributes['Kilroy'] = 'was here'
        f.close()
        f = netcdf_file(filename, 'a')
        assert_equal(f._attributes['Kilroy'], b'was here')
        f._attributes['naughty'] = b'Zoot'
        f.close()
        f = netcdf_file(filename, 'r')
        assert_equal(f._attributes['Kilroy'], b'was here')
        assert_equal(f._attributes['naughty'], b'Zoot')
        f.close()