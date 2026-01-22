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
def _test_move_file(self, protocol):
    other_fs = open_fs(protocol)
    text = 'Hello, World'
    self.fs.makedir('foo').writetext('test.txt', text)
    self.assert_text('foo/test.txt', text)
    with self.assertRaises(errors.ResourceNotFound):
        fs.move.move_file(self.fs, 'foo/test.txt', other_fs, 'foo/test2.txt')
    other_fs.makedir('foo')
    fs.move.move_file(self.fs, 'foo/test.txt', other_fs, 'foo/test2.txt')
    self.assertEqual(other_fs.readtext('foo/test2.txt'), text)