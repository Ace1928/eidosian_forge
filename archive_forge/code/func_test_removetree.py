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
def test_removetree(self):
    self.fs.makedirs('spam')
    self.fs.makedirs('foo/bar/baz')
    self.fs.makedirs('foo/egg')
    self.fs.makedirs('foo/a/b/c/d/e')
    self.fs.create('foo/egg.txt')
    self.fs.create('foo/bar/egg.bin')
    self.fs.create('foo/bar/baz/egg.txt')
    self.fs.create('foo/a/b/c/1.txt')
    self.fs.create('foo/a/b/c/2.txt')
    self.fs.create('foo/a/b/c/3.txt')
    self.assert_exists('foo/egg.txt')
    self.assert_exists('foo/bar/egg.bin')
    self.fs.removetree('foo')
    self.assert_not_exists('foo')
    self.assert_exists('spam')
    self.fs.create('bar')
    with self.assertRaises(errors.DirectoryExpected):
        self.fs.removetree('bar')
    with self.assertRaises(errors.ResourceNotFound):
        self.fs.removetree('foofoo')