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
class TestUserblock(TestCase):
    """
        Feature: Files can be create with user blocks
    """

    def test_create_blocksize(self):
        """ User blocks created with w, w-, x and properties work correctly """
        f = File(self.mktemp(), 'w-', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()
        f = File(self.mktemp(), 'x', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()
        f = File(self.mktemp(), 'w', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()
        with self.assertRaises(ValueError):
            File(self.mktemp(), 'w', userblock_size='non')

    def test_write_only(self):
        """ User block only allowed for write """
        name = self.mktemp()
        f = File(name, 'w')
        f.close()
        with self.assertRaises(ValueError):
            f = h5py.File(name, 'r', userblock_size=512)
        with self.assertRaises(ValueError):
            f = h5py.File(name, 'r+', userblock_size=512)

    def test_match_existing(self):
        """ User block size must match that of file when opening for append """
        name = self.mktemp()
        f = File(name, 'w', userblock_size=512)
        f.close()
        with self.assertRaises(ValueError):
            f = File(name, 'a', userblock_size=1024)
        f = File(name, 'a', userblock_size=512)
        try:
            self.assertEqual(f.userblock_size, 512)
        finally:
            f.close()

    def test_power_of_two(self):
        """ User block size must be a power of 2 and at least 512 """
        name = self.mktemp()
        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=128)
        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=513)
        with self.assertRaises(ValueError):
            f = File(name, 'w', userblock_size=1023)

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