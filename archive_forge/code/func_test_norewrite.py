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
@skipIf(not getattr(os, 'sync', None), 'Requires sync support')
def test_norewrite(self):
    repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        filea = Blob.from_string(b'file a')
        filea_path = os.path.join(repo_dir, 'a')
        tree = Tree()
        tree[b'a'] = (stat.S_IFREG | 420, filea.id)
        repo.object_store.add_objects([(o, None) for o in [filea, tree]])
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        os.sync()
        mtime = os.stat(filea_path).st_mtime
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        os.sync()
        self.assertEqual(mtime, os.stat(filea_path).st_mtime)
        with open(filea_path, 'wb') as fh:
            fh.write(b'test a')
        os.sync()
        mtime = os.stat(filea_path).st_mtime
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        os.sync()
        with open(filea_path, 'rb') as fh:
            self.assertEqual(b'file a', fh.read())