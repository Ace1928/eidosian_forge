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
def test_unicode_hdf5_python_consistent(self):
    """ Unicode filenames can be used, and seen correctly from python
        """
    fname = self.mktemp(prefix=chr(8218))
    with File(fname, 'w') as f:
        self.assertTrue(os.path.exists(fname))