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
def test_only_with_page_strategy(self):
    """Allow page buffering only with fs_strategy="page".
        """
    fname = self.mktemp()
    with File(fname, mode='w', fs_strategy='page', page_buf_size=16 * 1024):
        pass
    with self.assertRaises(OSError):
        File(fname, mode='w', page_buf_size=16 * 1024)
    with self.assertRaises(OSError):
        File(fname, mode='w', fs_strategy='fsm', page_buf_size=16 * 1024)
    with self.assertRaises(OSError):
        File(fname, mode='w', fs_strategy='aggregate', page_buf_size=16 * 1024)