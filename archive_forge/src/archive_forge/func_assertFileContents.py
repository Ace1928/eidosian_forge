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
def assertFileContents(self, path, contents, symlink=False):
    if symlink:
        self.assertEqual(os.readlink(path), contents)
    else:
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), contents)