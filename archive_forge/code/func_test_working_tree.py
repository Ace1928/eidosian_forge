import glob
import locale
import os
import shutil
import stat
import sys
import tempfile
import warnings
from dulwich import errors, objects, porcelain
from dulwich.tests import TestCase, skipIf
from ..config import Config
from ..errors import NotGitRepository
from ..object_store import tree_lookup_path
from ..repo import (
from .utils import open_repo, setup_warning_catcher, tear_down_repo
import sys
from dulwich.repo import Repo
def test_working_tree(self):
    temp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, temp_dir)
    worktree_temp_dir = tempfile.mkdtemp()
    self.addCleanup(shutil.rmtree, worktree_temp_dir)
    r = Repo.init(temp_dir)
    self.addCleanup(r.close)
    root_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    r.refs[b'refs/heads/master'] = root_sha
    w = Repo._init_new_working_directory(worktree_temp_dir, r)
    self.addCleanup(w.close)
    new_sha = w.do_commit(b'new commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    w.refs[b'HEAD'] = new_sha
    self.assertEqual(os.path.abspath(r.controldir()), os.path.abspath(w.commondir()))
    self.assertEqual(r.refs.keys(), w.refs.keys())
    self.assertNotEqual(r.head(), w.head())