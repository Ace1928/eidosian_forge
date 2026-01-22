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
class TestFileOpen(TestCase):
    """
        Feature: Opening files with Python-style modes.
    """

    def test_default(self):
        """ Default semantics in the presence or absence of a file """
        fname = self.mktemp()
        with pytest.raises(FileNotFoundError):
            with File(fname):
                pass
        with File(fname, 'w'):
            pass
        os.chmod(fname, stat.S_IREAD)
        try:
            with File(fname) as f:
                self.assertTrue(f)
                self.assertEqual(f.mode, 'r')
        finally:
            os.chmod(fname, stat.S_IWRITE)
        with open(fname, 'wb') as f:
            f.write(b'\x00')
        with self.assertRaises(OSError):
            File(fname)

    def test_create(self):
        """ Mode 'w' opens file in overwrite mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        self.assertTrue(fid)
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'w')
        self.assertNotIn('foo', fid)
        fid.close()

    def test_create_exclusive(self):
        """ Mode 'w-' opens file in exclusive mode """
        fname = self.mktemp()
        fid = File(fname, 'w-')
        self.assertTrue(fid)
        fid.close()
        with self.assertRaises(FileExistsError):
            File(fname, 'w-')

    def test_append(self):
        """ Mode 'a' opens file in append/readwrite mode, creating if necessary """
        fname = self.mktemp()
        fid = File(fname, 'a')
        try:
            self.assertTrue(fid)
            fid.create_group('foo')
            assert 'foo' in fid
        finally:
            fid.close()
        fid = File(fname, 'a')
        try:
            assert 'foo' in fid
            fid.create_group('bar')
            assert 'bar' in fid
        finally:
            fid.close()
        os.chmod(fname, stat.S_IREAD)
        try:
            with pytest.raises(PermissionError):
                File(fname, 'a')
        finally:
            os.chmod(fname, stat.S_IREAD | stat.S_IWRITE)

    def test_readonly(self):
        """ Mode 'r' opens file in readonly mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.close()
        self.assertFalse(fid)
        fid = File(fname, 'r')
        self.assertTrue(fid)
        with self.assertRaises(ValueError):
            fid.create_group('foo')
        fid.close()

    def test_readwrite(self):
        """ Mode 'r+' opens existing file in readwrite mode """
        fname = self.mktemp()
        fid = File(fname, 'w')
        fid.create_group('foo')
        fid.close()
        fid = File(fname, 'r+')
        assert 'foo' in fid
        fid.create_group('bar')
        assert 'bar' in fid
        fid.close()

    def test_nonexistent_file(self):
        """ Modes 'r' and 'r+' do not create files """
        fname = self.mktemp()
        with self.assertRaises(FileNotFoundError):
            File(fname, 'r')
        with self.assertRaises(FileNotFoundError):
            File(fname, 'r+')

    def test_invalid_mode(self):
        """ Invalid modes raise ValueError """
        with self.assertRaises(ValueError):
            File(self.mktemp(), 'mongoose')