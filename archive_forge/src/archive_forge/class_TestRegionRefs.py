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
class TestRegionRefs(BaseDataset):
    """
        Various features of region references
    """

    def setUp(self):
        BaseDataset.setUp(self)
        self.data = np.arange(100 * 100).reshape((100, 100))
        self.dset = self.f.create_dataset('x', data=self.data)
        self.dset[...] = self.data

    def test_create_ref(self):
        """ Region references can be used as slicing arguments """
        slic = np.s_[25:35, 10:100:5]
        ref = self.dset.regionref[slic]
        self.assertArrayEqual(self.dset[ref], self.data[slic])

    @empty_regionref_xfail
    def test_empty_region(self):
        ref = self.dset.regionref[:0]
        out = self.dset[ref]
        assert out.size == 0

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

    def test_ref_shape(self):
        """ Region reference shape and selection shape """
        slic = np.s_[25:35, 10:100:5]
        ref = self.dset.regionref[slic]
        self.assertEqual(self.dset.regionref.shape(ref), self.dset.shape)
        self.assertEqual(self.dset.regionref.selection(ref), (10, 18))