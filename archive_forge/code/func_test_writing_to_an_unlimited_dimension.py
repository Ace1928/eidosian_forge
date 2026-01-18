import gc
import io
import random
import re
import string
import tempfile
from os import environ as env
import h5py
import netCDF4
import numpy as np
import pytest
from packaging import version
from pytest import raises
import h5netcdf
from h5netcdf import legacyapi
from h5netcdf.core import NOT_A_VARIABLE, CompatibilityError
def test_writing_to_an_unlimited_dimension(tmp_local_or_remote_netcdf):
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'w') as f:
        f.dimensions['x'] = None
        f.dimensions['y'] = 3
        f.dimensions['z'] = None
        with pytest.raises(ValueError) as e:
            f.create_variable('dummy1', data=np.array([[1, 2, 3]]), dimensions=('x', 'y'))
            assert e.value.args[0] == 'Shape tuple is incompatible with data'
        f.create_variable('dummy1', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy2', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummy3', dimensions=('x', 'y'), dtype=np.int64)
        f.create_variable('dummyX', dimensions=('x', 'y', 'z'), dtype=np.int64)
        g = f.create_group('test')
        g.create_variable('dummy4', dimensions=('y', 'x', 'x'), dtype=np.int64)
        g.create_variable('dummy5', dimensions=('y', 'y'), dtype=np.int64)
        assert f.variables['dummy1'].shape == (0, 3)
        assert f.variables['dummy2'].shape == (0, 3)
        assert f.variables['dummy3'].shape == (0, 3)
        assert f.variables['dummyX'].shape == (0, 3, 0)
        assert g.variables['dummy4'].shape == (3, 0, 0)
        assert g.variables['dummy5'].shape == (3, 3)
        f.resize_dimension('x', 2)
        assert f.variables['dummy1'].shape == (2, 3)
        assert f.variables['dummy2'].shape == (2, 3)
        assert f.variables['dummy3'].shape == (2, 3)
        assert f.variables['dummyX'].shape == (2, 3, 0)
        assert g.variables['dummy4'].shape == (3, 2, 2)
        assert g.variables['dummy5'].shape == (3, 3)
        if tmp_local_or_remote_netcdf.startswith(remote_h5):
            expected_errors = pytest.raises(OSError)
        else:
            expected_errors = memoryview(b'')
        with expected_errors as e:
            f.variables['dummy3'][...] = [[1, 2, 3]]
            np.testing.assert_allclose(f.variables['dummy3'], [[1, 2, 3], [1, 2, 3]])
        if tmp_local_or_remote_netcdf.startswith(remote_h5):
            assert 'Got asyncio.IncompleteReadError' in e.value.args[0]