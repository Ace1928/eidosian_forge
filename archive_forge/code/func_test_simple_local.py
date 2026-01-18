import contextlib
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from io import BytesIO, StringIO
from unittest import skipIf
from dulwich import porcelain
from dulwich.tests import TestCase
from ..diff_tree import tree_changes
from ..errors import CommitError
from ..objects import ZERO_SHA, Blob, Tag, Tree
from ..porcelain import CheckoutError
from ..repo import NoIndexPresent, Repo
from ..server import DictBackend
from ..web import make_server, make_wsgi_chain
from .utils import build_commit_graph, make_commit, make_object
def test_simple_local(self):
    f1_1 = make_object(Blob, data=b'f1')
    commit_spec = [[1], [2, 1], [3, 1, 2]]
    trees = {1: [(b'f1', f1_1), (b'f2', f1_1)], 2: [(b'f1', f1_1), (b'f2', f1_1)], 3: [(b'f1', f1_1), (b'f2', f1_1)]}
    c1, c2, c3 = build_commit_graph(self.repo.object_store, commit_spec, trees)
    self.repo.refs[b'refs/heads/master'] = c3.id
    self.repo.refs[b'refs/tags/foo'] = c3.id
    target_path = tempfile.mkdtemp()
    errstream = BytesIO()
    self.addCleanup(shutil.rmtree, target_path)
    r = porcelain.clone(self.repo.path, target_path, checkout=False, errstream=errstream)
    self.addCleanup(r.close)
    self.assertEqual(r.path, target_path)
    target_repo = Repo(target_path)
    self.assertEqual(0, len(target_repo.open_index()))
    self.assertEqual(c3.id, target_repo.refs[b'refs/tags/foo'])
    self.assertNotIn(b'f1', os.listdir(target_path))
    self.assertNotIn(b'f2', os.listdir(target_path))
    c = r.get_config()
    encoded_path = self.repo.path
    if not isinstance(encoded_path, bytes):
        encoded_path = encoded_path.encode('utf-8')
    self.assertEqual(encoded_path, c.get((b'remote', b'origin'), b'url'))
    self.assertEqual(b'+refs/heads/*:refs/remotes/origin/*', c.get((b'remote', b'origin'), b'fetch'))