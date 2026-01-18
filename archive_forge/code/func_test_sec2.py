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
@ut.skipUnless(os.name == 'posix', 'Sec2 driver is supported on posix')
def test_sec2(self):
    """ Sec2 driver is supported on posix """
    fid = File(self.mktemp(), 'w', driver='sec2')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'sec2')
    fid.close()
    fid = File(self.mktemp(), 'a', driver='sec2')
    self.assertTrue(fid)
    self.assertEqual(fid.driver, 'sec2')
    fid.close()