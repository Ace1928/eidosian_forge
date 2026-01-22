import os
import shutil
import stat
import struct
import sys
import tempfile
from io import BytesIO
from dulwich.tests import TestCase, skipIf
from ..index import (
from ..object_store import MemoryObjectStore
from ..objects import S_IFGITLINK, Blob, Commit, Tree
from ..repo import Repo
class SimpleIndexTestCase(IndexTestCase):

    def test_len(self):
        self.assertEqual(1, len(self.get_simple_index('index')))

    def test_iter(self):
        self.assertEqual([b'bla'], list(self.get_simple_index('index')))

    def test_iterobjects(self):
        self.assertEqual([(b'bla', b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391', 33188)], list(self.get_simple_index('index').iterobjects()))

    def test_getitem(self):
        self.assertEqual(IndexEntry((1230680220, 0), (1230680220, 0), 2050, 3761020, 33188, 1000, 1000, 0, b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391'), self.get_simple_index('index')[b'bla'])

    def test_empty(self):
        i = self.get_simple_index('notanindex')
        self.assertEqual(0, len(i))
        self.assertFalse(os.path.exists(i._filename))

    def test_against_empty_tree(self):
        i = self.get_simple_index('index')
        changes = list(i.changes_from_tree(MemoryObjectStore(), None))
        self.assertEqual(1, len(changes))
        (oldname, newname), (oldmode, newmode), (oldsha, newsha) = changes[0]
        self.assertEqual(b'bla', newname)
        self.assertEqual(b'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391', newsha)