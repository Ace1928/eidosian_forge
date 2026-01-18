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
def test_nonempty(self):
    repo_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    with Repo.init(repo_dir) as repo:
        filea = Blob.from_string(b'file a')
        fileb = Blob.from_string(b'file b')
        filed = Blob.from_string(b'file d')
        tree = Tree()
        tree[b'a'] = (stat.S_IFREG | 420, filea.id)
        tree[b'b'] = (stat.S_IFREG | 420, fileb.id)
        tree[b'c/d'] = (stat.S_IFREG | 420, filed.id)
        repo.object_store.add_objects([(o, None) for o in [filea, fileb, filed, tree]])
        build_index_from_tree(repo.path, repo.index_path(), repo.object_store, tree.id)
        index = repo.open_index()
        self.assertEqual(len(index), 3)
        apath = os.path.join(repo.path, 'a')
        self.assertTrue(os.path.exists(apath))
        self.assertReasonableIndexEntry(index[b'a'], stat.S_IFREG | 420, 6, filea.id)
        self.assertFileContents(apath, b'file a')
        bpath = os.path.join(repo.path, 'b')
        self.assertTrue(os.path.exists(bpath))
        self.assertReasonableIndexEntry(index[b'b'], stat.S_IFREG | 420, 6, fileb.id)
        self.assertFileContents(bpath, b'file b')
        dpath = os.path.join(repo.path, 'c', 'd')
        self.assertTrue(os.path.exists(dpath))
        self.assertReasonableIndexEntry(index[b'c/d'], stat.S_IFREG | 420, 6, filed.id)
        self.assertFileContents(dpath, b'file d')
        self.assertEqual(['.git', 'a', 'b', 'c'], sorted(os.listdir(repo.path)))
        self.assertEqual(['d'], sorted(os.listdir(os.path.join(repo.path, 'c'))))