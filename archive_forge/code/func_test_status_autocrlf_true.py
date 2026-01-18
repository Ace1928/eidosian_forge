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
def test_status_autocrlf_true(self):
    file_path = os.path.join(self.repo.path, 'crlf')
    with open(file_path, 'wb') as f:
        f.write(b'line1\nline2')
    porcelain.add(repo=self.repo.path, paths=[file_path])
    porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
    with open(file_path, 'wb') as f:
        f.write(b'line1\r\nline2')
    c = self.repo.get_config()
    c.set('core', 'autocrlf', True)
    c.write_to_path()
    results = porcelain.status(self.repo)
    self.assertDictEqual({'add': [], 'delete': [], 'modify': []}, results.staged)
    self.assertListEqual(results.unstaged, [])
    self.assertListEqual(results.untracked, [])