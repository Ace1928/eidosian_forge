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
def test_write_only(self):
    """ User block only allowed for write """
    name = self.mktemp()
    f = File(name, 'w')
    f.close()
    with self.assertRaises(ValueError):
        f = h5py.File(name, 'r', userblock_size=512)
    with self.assertRaises(ValueError):
        f = h5py.File(name, 'r+', userblock_size=512)