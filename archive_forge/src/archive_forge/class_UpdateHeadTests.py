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
class UpdateHeadTests(PorcelainTestCase):

    def test_set_to_branch(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo.refs[b'refs/heads/blah'] = c1.id
        porcelain.update_head(self.repo, 'blah')
        self.assertEqual(c1.id, self.repo.head())
        self.assertEqual(b'ref: refs/heads/blah', self.repo.refs.read_ref(b'HEAD'))

    def test_set_to_branch_detached(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo.refs[b'refs/heads/blah'] = c1.id
        porcelain.update_head(self.repo, 'blah', detached=True)
        self.assertEqual(c1.id, self.repo.head())
        self.assertEqual(c1.id, self.repo.refs.read_ref(b'HEAD'))

    def test_set_to_commit_detached(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo.refs[b'refs/heads/blah'] = c1.id
        porcelain.update_head(self.repo, c1.id, detached=True)
        self.assertEqual(c1.id, self.repo.head())
        self.assertEqual(c1.id, self.repo.refs.read_ref(b'HEAD'))

    def test_set_new_branch(self):
        [c1] = build_commit_graph(self.repo.object_store, [[1]])
        self.repo.refs[b'refs/heads/blah'] = c1.id
        porcelain.update_head(self.repo, 'blah', new_branch='bar')
        self.assertEqual(c1.id, self.repo.head())
        self.assertEqual(b'ref: refs/heads/bar', self.repo.refs.read_ref(b'HEAD'))