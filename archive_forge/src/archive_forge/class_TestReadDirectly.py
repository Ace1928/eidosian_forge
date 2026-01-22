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
class TestReadDirectly:
    """
        Feature: Read data directly from Dataset into a Numpy array
    """

    @pytest.mark.parametrize('source_shape,dest_shape,source_sel,dest_sel', [((100,), (100,), np.s_[0:10], np.s_[50:60]), ((70,), (100,), np.s_[50:60], np.s_[90:]), ((30, 10), (20, 20), np.s_[:20, :], np.s_[:, :10]), ((5, 7, 9), (6,), np.s_[2, :6, 3], np.s_[:])])
    def test_read_direct(self, writable_file, source_shape, dest_shape, source_sel, dest_sel):
        source_values = np.arange(product(source_shape), dtype='int64').reshape(source_shape)
        dset = writable_file.create_dataset('dset', source_shape, data=source_values)
        arr = np.full(dest_shape, -1, dtype='int64')
        expected = arr.copy()
        expected[dest_sel] = source_values[source_sel]
        dset.read_direct(arr, source_sel, dest_sel)
        np.testing.assert_array_equal(arr, expected)

    def test_no_sel(self, writable_file):
        dset = writable_file.create_dataset('dset', (10,), data=np.arange(10, dtype='int64'))
        arr = np.ones((10,), dtype='int64')
        dset.read_direct(arr)
        np.testing.assert_array_equal(arr, np.arange(10, dtype='int64'))

    def test_empty(self, writable_file):
        empty_dset = writable_file.create_dataset('edset', dtype='int64')
        arr = np.ones((100,), 'int64')
        with pytest.raises(TypeError):
            empty_dset.read_direct(arr, np.s_[0:10], np.s_[50:60])

    def test_wrong_shape(self, writable_file):
        dset = writable_file.create_dataset('dset', (100,), dtype='int64')
        arr = np.ones((200,))
        with pytest.raises(TypeError):
            dset.read_direct(arr)

    def test_not_c_contiguous(self, writable_file):
        dset = writable_file.create_dataset('dset', (10, 10), dtype='int64')
        arr = np.ones((10, 10), order='F')
        with pytest.raises(TypeError):
            dset.read_direct(arr)