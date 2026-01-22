import pytest
import os
import stat
import pickle
import tempfile
import subprocess
import sys
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
from h5py._hl.files import direct_vfd
from h5py import File
import h5py
from .. import h5
import pathlib
import sys
import h5py
class TestNewLibver(TestCase):
    """
        Feature: File format compatibility bounds can be specified when
        opening a file.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if h5py.version.hdf5_version_tuple < (1, 11, 4):
            cls.latest = 'v110'
        elif h5py.version.hdf5_version_tuple < (1, 13, 0):
            cls.latest = 'v112'
        else:
            cls.latest = 'v114'

    def test_default(self):
        """ Opening with no libver arg """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', self.latest))
        f.close()

    def test_single(self):
        """ Opening with single libver arg """
        f = File(self.mktemp(), 'w', libver='latest')
        self.assertEqual(f.libver, (self.latest, self.latest))
        f.close()

    def test_single_v108(self):
        """ Opening with "v108" libver arg """
        f = File(self.mktemp(), 'w', libver='v108')
        self.assertEqual(f.libver, ('v108', self.latest))
        f.close()

    def test_single_v110(self):
        """ Opening with "v110" libver arg """
        f = File(self.mktemp(), 'w', libver='v110')
        self.assertEqual(f.libver, ('v110', self.latest))
        f.close()

    @ut.skipIf(h5py.version.hdf5_version_tuple < (1, 11, 4), 'Requires HDF5 1.11.4 or later')
    def test_single_v112(self):
        """ Opening with "v112" libver arg """
        f = File(self.mktemp(), 'w', libver='v112')
        self.assertEqual(f.libver, ('v112', self.latest))
        f.close()

    def test_multiple(self):
        """ Opening with two libver args """
        f = File(self.mktemp(), 'w', libver=('earliest', 'v108'))
        self.assertEqual(f.libver, ('earliest', 'v108'))
        f.close()

    def test_none(self):
        """ Omitting libver arg results in maximum compatibility """
        f = File(self.mktemp(), 'w')
        self.assertEqual(f.libver, ('earliest', self.latest))
        f.close()