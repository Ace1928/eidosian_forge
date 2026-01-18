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
def test_tree_to_fs_path(self):
    tree_path = 'délwíçh/foo'.encode()
    fs_path = _tree_to_fs_path(b'/prefix/path', tree_path)
    self.assertEqual(fs_path, os.fsencode(os.path.join('/prefix/path', 'délwíçh', 'foo')))