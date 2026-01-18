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
@pytest.mark.skipif(version.parse(h5py.__version__) < version.parse('3.7.0'), reason='does not work with h5py < 3.7.0')
def test_vlen_string_dataset_fillvalue(tmp_local_netcdf, decode_vlen_strings):
    with h5netcdf.File(tmp_local_netcdf, 'w') as ds:
        ds.dimensions = {'string': 10}
        dt0 = h5py.string_dtype()
        fill_value0 = 'bár'
        ds.create_variable('x0', ('string',), dtype=dt0, fillvalue=fill_value0)
        dt1 = h5py.string_dtype('ascii')
        fill_value1 = 'bar'
        ds.create_variable('x1', ('string',), dtype=dt1, fillvalue=fill_value1)
    with h5netcdf.File(tmp_local_netcdf, 'r', **decode_vlen_strings) as ds:
        decode_vlen = decode_vlen_strings['decode_vlen_strings']
        fvalue0 = fill_value0 if decode_vlen else fill_value0.encode('utf-8')
        fvalue1 = fill_value1 if decode_vlen else fill_value1.encode('utf-8')
        assert ds['x0'][0] == fvalue0
        assert ds['x0'].attrs['_FillValue'] == fill_value0
        assert ds['x1'][0] == fvalue1
        assert ds['x1'].attrs['_FillValue'] == fill_value1
    with legacyapi.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['x0'][0] == fill_value0
        assert ds['x0']._FillValue == fill_value0
        assert ds['x1'][0] == fill_value1
        assert ds['x1']._FillValue == fill_value1
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['x0'][0] == fill_value0
        assert ds['x0']._FillValue == fill_value0
        assert ds['x1'][0] == fill_value1
        assert ds['x1']._FillValue == fill_value1
    with legacyapi.Dataset(tmp_local_netcdf, 'w') as ds:
        ds.createDimension('string', 10)
        fill_value0 = 'bár'
        ds.createVariable('x0', str, ('string',), fill_value=fill_value0)
        fill_value1 = 'bar'
        ds.createVariable('x1', str, ('string',), fill_value=fill_value1)
    with h5netcdf.File(tmp_local_netcdf, 'r', **decode_vlen_strings) as ds:
        decode_vlen = decode_vlen_strings['decode_vlen_strings']
        fvalue0 = fill_value0 if decode_vlen else fill_value0.encode('utf-8')
        fvalue1 = fill_value1 if decode_vlen else fill_value1.encode('utf-8')
        assert ds['x0'][0] == fvalue0
        assert ds['x0'].attrs['_FillValue'] == fill_value0
        assert ds['x1'][0] == fvalue1
        assert ds['x1'].attrs['_FillValue'] == fill_value1
    with legacyapi.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['x0'][0] == fill_value0
        assert ds['x0']._FillValue == fill_value0
        assert ds['x1'][0] == fill_value1
        assert ds['x1']._FillValue == fill_value1
    with netCDF4.Dataset(tmp_local_netcdf, 'r') as ds:
        assert ds['x0'][0] == fill_value0
        assert ds['x0']._FillValue == fill_value0
        assert ds['x1'][0] == fill_value1
        assert ds['x1']._FillValue == fill_value1