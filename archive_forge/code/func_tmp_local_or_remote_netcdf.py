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
@pytest.fixture(params=['testfile.nc', 'hdf5://testfile'])
def tmp_local_or_remote_netcdf(request, tmpdir, hsds_up):
    if request.param.startswith(remote_h5):
        if without_h5pyd:
            pytest.skip('h5pyd package not available')
        elif not hsds_up:
            pytest.skip('HSDS service not running')
        rnd = ''.join((random.choice(string.ascii_uppercase) for _ in range(5)))
        return 'hdf5://' + 'home' + '/' + env['HS_USERNAME'] + '/' + 'testfile' + rnd + '.nc'
    else:
        return str(tmpdir.join(request.param))