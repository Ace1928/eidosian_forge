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
def test_read_write_files():
    cwd = os.getcwd()
    try:
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        with make_simple('simple.nc', 'w') as f:
            pass
        with netcdf_file('simple.nc', 'a') as f:
            check_simple(f)
            f._attributes['appendRan'] = 1
        with netcdf_file('simple.nc') as f:
            assert_equal(f.use_mmap, not IS_PYPY)
            check_simple(f)
            assert_equal(f._attributes['appendRan'], 1)
        with netcdf_file('simple.nc', 'a') as f:
            assert_(not f.use_mmap)
            check_simple(f)
            assert_equal(f._attributes['appendRan'], 1)
        with netcdf_file('simple.nc', mmap=False) as f:
            assert_(not f.use_mmap)
            check_simple(f)
        with open('simple.nc', 'rb') as fobj:
            with netcdf_file(fobj) as f:
                assert_(not f.use_mmap)
                check_simple(f)
        with suppress_warnings() as sup:
            if IS_PYPY:
                sup.filter(RuntimeWarning, 'Cannot close a netcdf_file opened with mmap=True.*')
            with open('simple.nc', 'rb') as fobj:
                with netcdf_file(fobj, mmap=True) as f:
                    assert_(f.use_mmap)
                    check_simple(f)
        with open('simple.nc', 'r+b') as fobj:
            with netcdf_file(fobj, 'a') as f:
                assert_(not f.use_mmap)
                check_simple(f)
                f.createDimension('app_dim', 1)
                var = f.createVariable('app_var', 'i', ('app_dim',))
                var[:] = 42
        with netcdf_file('simple.nc') as f:
            check_simple(f)
            assert_equal(f.variables['app_var'][:], 42)
    finally:
        if IS_PYPY:
            break_cycles()
            break_cycles()
        os.chdir(cwd)
        shutil.rmtree(tmpdir)