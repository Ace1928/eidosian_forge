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
def test_write_block(self):
    """ Test that writing to a user block does not destroy the file """
    name = self.mktemp()
    f = File(name, 'w', userblock_size=512)
    f.create_group('Foobar')
    f.close()
    pyfile = open(name, 'r+b')
    try:
        pyfile.write(b'X' * 512)
    finally:
        pyfile.close()
    f = h5py.File(name, 'r')
    try:
        assert 'Foobar' in f
    finally:
        f.close()
    pyfile = open(name, 'rb')
    try:
        self.assertEqual(pyfile.read(512), b'X' * 512)
    finally:
        pyfile.close()