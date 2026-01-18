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
@pytest.mark.parametrize('source_shape,dest_shape,source_sel,dest_sel', [((100,), (100,), np.s_[0:10], np.s_[50:60]), ((70,), (100,), np.s_[50:60], np.s_[90:]), ((30, 10), (20, 20), np.s_[:20, :], np.s_[:, :10]), ((5, 7, 9), (6,), np.s_[2, :6, 3], np.s_[:])])
def test_write_direct(self, writable_file, source_shape, dest_shape, source_sel, dest_sel):
    dset = writable_file.create_dataset('dset', dest_shape, dtype='int32', fillvalue=-1)
    arr = np.arange(product(source_shape)).reshape(source_shape)
    expected = np.full(dest_shape, -1, dtype='int32')
    expected[dest_sel] = arr[source_sel]
    dset.write_direct(arr, source_sel, dest_sel)
    np.testing.assert_array_equal(dset[:], expected)