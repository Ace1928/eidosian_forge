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
@empty_regionref_xfail
def test_scalar_dataset(self):
    ds = self.f.create_dataset('scalar', data=1.0, dtype='f4')
    sid = h5py.h5s.create(h5py.h5s.SCALAR)
    sid.select_none()
    ref = h5py.h5r.create(ds.id, b'.', h5py.h5r.DATASET_REGION, sid)
    assert ds[ref] == h5py.Empty(np.dtype('f4'))
    sid.select_all()
    ref = h5py.h5r.create(ds.id, b'.', h5py.h5r.DATASET_REGION, sid)
    assert ds[ref] == ds[()]