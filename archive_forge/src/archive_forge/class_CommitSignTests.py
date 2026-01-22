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
@skipIf(platform.python_implementation() == 'PyPy' or sys.platform == 'win32', 'gpgme not easily available or supported on Windows and PyPy')
class CommitSignTests(PorcelainGpgTestCase):

    def test_default_key(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        cfg = self.repo.get_config()
        cfg.set(('user',), 'signingKey', PorcelainGpgTestCase.DEFAULT_KEY_ID)
        self.import_default_key()
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>', signoff=True)
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)
        commit = self.repo.get_object(sha)
        commit.verify()
        commit.verify(keyids=[PorcelainGpgTestCase.DEFAULT_KEY_ID])
        self.import_non_default_key()
        self.assertRaises(gpg.errors.MissingSignatures, commit.verify, keyids=[PorcelainGpgTestCase.NON_DEFAULT_KEY_ID])
        commit.committer = b'Alice <alice@example.com>'
        self.assertRaises(gpg.errors.BadSignatures, commit.verify)

    def test_non_default_key(self):
        c1, c2, c3 = build_commit_graph(self.repo.object_store, [[1], [2, 1], [3, 1, 2]])
        self.repo.refs[b'HEAD'] = c3.id
        cfg = self.repo.get_config()
        cfg.set(('user',), 'signingKey', PorcelainGpgTestCase.DEFAULT_KEY_ID)
        self.import_non_default_key()
        sha = porcelain.commit(self.repo.path, message='Some message', author='Joe <joe@example.com>', committer='Bob <bob@example.com>', signoff=PorcelainGpgTestCase.NON_DEFAULT_KEY_ID)
        self.assertIsInstance(sha, bytes)
        self.assertEqual(len(sha), 40)
        commit = self.repo.get_object(sha)
        commit.verify()