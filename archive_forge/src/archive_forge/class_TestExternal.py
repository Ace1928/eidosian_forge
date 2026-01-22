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
class TestExternal(BaseDataset):
    """
        Feature: Datasets with the external storage property
    """

    def test_contents(self):
        """ Create and access an external dataset """
        shape = (6, 100)
        testdata = np.random.random(shape)
        ext_file = self.mktemp()
        external = [(ext_file, 0, h5f.UNLIMITED)]
        dset = self.f.create_dataset('foo', shape, dtype=testdata.dtype, external=external, efile_prefix='${ORIGIN}')
        dset[...] = testdata
        assert dset.external is not None
        with open(ext_file, 'rb') as fid:
            contents = fid.read()
        assert contents == testdata.tobytes()
        efile_prefix = pathlib.Path(dset.id.get_access_plist().get_efile_prefix().decode()).as_posix()
        parent = pathlib.Path(self.f.filename).parent.as_posix()
        assert efile_prefix == parent

    def test_contents_efile_prefix(self):
        """ Create and access an external dataset using an efile_prefix"""
        shape = (6, 100)
        testdata = np.random.random(shape)
        ext_file = self.mktemp()
        external = [(os.path.basename(ext_file), 0, h5f.UNLIMITED)]
        dset = self.f.create_dataset('foo', shape, dtype=testdata.dtype, external=external, efile_prefix=os.path.dirname(ext_file))
        dset[...] = testdata
        assert dset.external is not None
        with open(ext_file, 'rb') as fid:
            contents = fid.read()
        assert contents == testdata.tobytes()
        if h5py.version.hdf5_version_tuple >= (1, 10, 0):
            efile_prefix = pathlib.Path(dset.id.get_access_plist().get_efile_prefix().decode()).as_posix()
            parent = pathlib.Path(ext_file).parent.as_posix()
            assert efile_prefix == parent
        dset2 = self.f.require_dataset('foo', shape, testdata.dtype, efile_prefix=os.path.dirname(ext_file))
        assert dset2.external is not None
        dset2[()] == testdata

    def test_name_str(self):
        """ External argument may be a file name str only """
        self.f.create_dataset('foo', (6, 100), external=self.mktemp())

    def test_name_path(self):
        """ External argument may be a file name path only """
        self.f.create_dataset('foo', (6, 100), external=pathlib.Path(self.mktemp()))

    def test_iter_multi(self):
        """ External argument may be an iterable of multiple tuples """
        ext_file = self.mktemp()
        N = 100
        external = iter(((ext_file, x * 1000, 1000) for x in range(N)))
        dset = self.f.create_dataset('poo', (6, 100), external=external)
        assert len(dset.external) == N

    def test_invalid(self):
        """ Test with invalid external lists """
        shape = (6, 100)
        ext_file = self.mktemp()
        for exc_type, external in [(TypeError, [ext_file]), (TypeError, [ext_file, 0]), (TypeError, [ext_file, 0, h5f.UNLIMITED]), (ValueError, [(ext_file,)]), (ValueError, [(ext_file, 0)]), (ValueError, [(ext_file, 0, h5f.UNLIMITED, 0)]), (TypeError, [(ext_file, 0, 'h5f.UNLIMITED')])]:
            with self.assertRaises(exc_type):
                self.f.create_dataset('foo', shape, external=external)