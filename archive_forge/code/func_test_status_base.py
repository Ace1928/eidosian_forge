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
def test_status_base(self):
    """Integration test for `status` functionality."""
    fullpath = os.path.join(self.repo.path, 'foo')
    with open(fullpath, 'w') as f:
        f.write('origstuff')
    porcelain.add(repo=self.repo.path, paths=[fullpath])
    porcelain.commit(repo=self.repo.path, message=b'test status', author=b'author <email>', committer=b'committer <email>')
    os.utime(fullpath, (0, 0))
    with open(fullpath, 'wb') as f:
        f.write(b'stuff')
    filename_add = 'bar'
    fullpath = os.path.join(self.repo.path, filename_add)
    with open(fullpath, 'w') as f:
        f.write('stuff')
    porcelain.add(repo=self.repo.path, paths=fullpath)
    results = porcelain.status(self.repo)
    self.assertEqual(results.staged['add'][0], filename_add.encode('ascii'))
    self.assertEqual(results.unstaged, [b'foo'])