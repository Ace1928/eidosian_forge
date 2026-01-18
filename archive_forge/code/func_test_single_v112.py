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
@ut.skipIf(h5py.version.hdf5_version_tuple < (1, 11, 4), 'Requires HDF5 1.11.4 or later')
def test_single_v112(self):
    """ Opening with "v112" libver arg """
    f = File(self.mktemp(), 'w', libver='v112')
    self.assertEqual(f.libver, ('v112', self.latest))
    f.close()