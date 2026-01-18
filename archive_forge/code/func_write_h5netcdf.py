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
def write_h5netcdf(tmp_netcdf):
    ds = h5netcdf.File(tmp_netcdf, 'w')
    ds.attrs['global'] = 42
    ds.attrs['other_attr'] = 'yes'
    ds.dimensions = {'x': 4, 'y': 5, 'z': 6, 'empty': 0, 'unlimited': None}
    v = ds.create_variable('foo', ('x', 'y'), float, chunks=(4, 5), compression='gzip', shuffle=True)
    v[...] = 1
    v.attrs['units'] = 'meters'
    remote_file = isinstance(tmp_netcdf, str) and tmp_netcdf.startswith(remote_h5)
    if not remote_file:
        v = ds.create_variable('y', ('y',), int, fillvalue=-1)
        v[:4] = np.arange(4)
    v = ds.create_variable('z', ('z', 'string3'), data=_char_array, fillvalue=b'X')
    v = ds.create_variable('scalar', data=np.float32(2.0))
    v = ds.create_variable('intscalar', data=np.int64(2))
    v = ds.create_variable('foo_unlimited', ('x', 'unlimited'), float)
    v[...] = 1
    with raises((h5netcdf.CompatibilityError, TypeError)):
        ds.create_variable('boolean', data=True)
    g = ds.create_group('subgroup')
    v = g.create_variable('subvar', ('x',), np.int32)
    v[...] = np.arange(4.0)
    with raises(AttributeError):
        v.attrs['_Netcdf4Dimid'] = -1
    g.dimensions['y'] = 10
    g.create_variable('y_var', ('y',), float)
    g.flush()
    ds.dimensions['mismatched_dim'] = 1
    ds.create_variable('mismatched_dim', dtype=int)
    ds.flush()
    dt = h5py.special_dtype(vlen=str)
    v = ds.create_variable('var_len_str', ('x',), dtype=dt)
    v[0] = _vlen_string
    ds.close()