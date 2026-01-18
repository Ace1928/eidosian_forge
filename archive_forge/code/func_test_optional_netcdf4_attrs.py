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
def test_optional_netcdf4_attrs(tmp_local_or_remote_netcdf):
    h5 = get_hdf5_module(tmp_local_or_remote_netcdf)
    with h5.File(tmp_local_or_remote_netcdf, 'w') as f:
        foo_data = np.arange(50).reshape(5, 10)
        f.create_dataset('foo', data=foo_data)
        f.create_dataset('x', data=np.arange(5))
        f.create_dataset('y', data=np.arange(10))
        f['x'].make_scale()
        f['y'].make_scale()
        f['foo'].dims[0].attach_scale(f['x'])
        f['foo'].dims[1].attach_scale(f['y'])
    with h5netcdf.File(tmp_local_or_remote_netcdf, 'r') as ds:
        assert ds['foo'].dimensions == ('x', 'y')
        assert ds.dimensions.keys() == {'x', 'y'}
        assert ds.dimensions['x'].size == 5
        assert ds.dimensions['y'].size == 10
        assert array_equal(ds['foo'], foo_data)