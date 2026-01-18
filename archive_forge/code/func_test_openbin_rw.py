from __future__ import absolute_import, unicode_literals
import io
import itertools
import json
import os
import six
import time
import unittest
import warnings
from datetime import datetime
from six import text_type
import fs.copy
import fs.move
from fs import ResourceType, Seek, errors, glob, walk
from fs.opener import open_fs
from fs.subfs import ClosingSubFS, SubFS
def test_openbin_rw(self):
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('doesnotexist', 'r')
    self.fs.makedir('foo')
    text = b'Hello, World\n'
    with self.fs.openbin('foo/hello', 'w') as f:
        repr(f)
        self.assertIn('b', f.mode)
        self.assertIsInstance(f, io.IOBase)
        self.assertTrue(f.writable())
        self.assertFalse(f.readable())
        self.assertEqual(len(text), f.write(text))
        self.assertFalse(f.closed)
    self.assertTrue(f.closed)
    with self.assertRaises(errors.FileExists):
        with self.fs.openbin('foo/hello', 'x') as f:
            pass
    with self.fs.openbin('foo/hello', 'r') as f:
        self.assertIn('b', f.mode)
        self.assertIsInstance(f, io.IOBase)
        self.assertTrue(f.readable())
        self.assertFalse(f.writable())
        hello = f.read()
        self.assertFalse(f.closed)
    self.assertTrue(f.closed)
    self.assertEqual(hello, text)
    self.assert_bytes('foo/hello', text)
    text = b'Goodbye, World'
    with self.fs.openbin('foo/hello', 'w') as f:
        self.assertEqual(len(text), f.write(text))
    self.assert_bytes('foo/hello', text)
    with self.assertRaises(errors.FileExpected):
        self.fs.openbin('foo')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.openbin('/foo/bar/test.txt')
    with self.fs.openbin('foo/hello') as f:
        try:
            fn = f.fileno()
        except io.UnsupportedOperation:
            pass
        else:
            self.assertEqual(os.read(fn, 7), b'Goodbye')
    lines = b'\n'.join([b'Line 1', b'Line 2', b'Line 3'])
    self.fs.writebytes('iter.bin', lines)
    with self.fs.openbin('iter.bin') as f:
        for actual, expected in zip(f, lines.splitlines(1)):
            self.assertEqual(actual, expected)