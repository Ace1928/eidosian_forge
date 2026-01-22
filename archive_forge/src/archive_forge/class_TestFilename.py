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
class TestFilename(TestCase):
    """
        Feature: The name of a File object can be retrieved via .filename
    """

    def test_filename(self):
        """ .filename behaves properly for string data """
        fname = self.mktemp()
        fid = File(fname, 'w')
        try:
            self.assertEqual(fid.filename, fname)
            self.assertIsInstance(fid.filename, str)
        finally:
            fid.close()