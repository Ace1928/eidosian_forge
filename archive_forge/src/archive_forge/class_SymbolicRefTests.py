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
class SymbolicRefTests(PorcelainTestCase):

    def test_set_wrong_symbolic_ref(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        self.assertRaises(porcelain.Error, porcelain.symbolic_ref, self.repo.path, b'foobar')

    def test_set_force_wrong_symbolic_ref(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        porcelain.symbolic_ref(self.repo.path, b'force_foobar', force=True)
        with self.repo.get_named_file('HEAD') as f:
            new_ref = f.read()
        self.assertEqual(new_ref, b'ref: refs/heads/force_foobar\n')

    def test_set_symbolic_ref(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        porcelain.symbolic_ref(self.repo.path, b'master')

    def test_set_symbolic_ref_other_than_master(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]], attrs=dict(refs='develop'))
        self.repo.refs[b'HEAD'] = c3.id
        self.repo.refs[b'refs/heads/develop'] = c3.id
        porcelain.symbolic_ref(self.repo.path, b'develop')
        with self.repo.get_named_file('HEAD') as f:
            new_ref = f.read()
        self.assertEqual(new_ref, b'ref: refs/heads/develop\n')