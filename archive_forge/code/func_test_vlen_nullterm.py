import pathlib
import os
import sys
import numpy as np
import platform
import pytest
import warnings
from .common import ut, TestCase
from .data_files import get_data_file_path
from h5py import File, Group, Dataset
from h5py._hl.base import is_empty_dataspace, product
from h5py import h5f, h5t
from h5py.h5py_warnings import H5pyDeprecationWarning
from h5py import version
import h5py
import h5py._hl.selections as sel
def test_vlen_nullterm():
    with File(get_data_file_path('vlen_string_dset_utc.h5')) as f:
        assert f['ds1'][0] == b'2009-12-20T10:16:18.662409Z'