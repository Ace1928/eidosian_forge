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
def test_unsupported_locking(self, tmp_path):
    """Test with erroneous file locking value"""
    fname = tmp_path / 'test.h5'
    with pytest.raises(ValueError):
        with h5py.File(fname, mode='r', locking='unsupported-value') as h5f_read:
            pass