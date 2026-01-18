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
def test_reading_special_datatype_created_with_c_api(tmp_local_netcdf):
    """Test reading a file with unsupported Datatype"""
    with netCDF4.Dataset(tmp_local_netcdf, 'w') as f:
        complex128 = np.dtype([('real', np.float64), ('imag', np.float64)])
        f.createCompoundType(complex128, 'complex128')
    with h5netcdf.File(tmp_local_netcdf, 'r') as f:
        pass