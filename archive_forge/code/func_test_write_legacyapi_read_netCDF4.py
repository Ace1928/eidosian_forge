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
def test_write_legacyapi_read_netCDF4(tmp_local_netcdf):
    roundtrip_legacy_netcdf(tmp_local_netcdf, netCDF4, legacyapi)