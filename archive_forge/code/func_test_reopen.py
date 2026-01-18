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
def test_reopen(self, tmp_path):
    """Test file locking when opening twice the same file"""
    fname = tmp_path / 'test.h5'
    with h5py.File(fname, mode='w', locking=True) as f:
        f.flush()
        with pytest.raises(OSError):
            with h5py.File(fname, mode='r', locking=False) as h5f_read:
                pass
        with h5py.File(fname, mode='r', locking=True) as h5f_read:
            pass
        with h5py.File(fname, mode='r', locking='best-effort') as h5f_read:
            pass