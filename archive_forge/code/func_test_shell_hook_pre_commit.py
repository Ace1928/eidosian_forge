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
def test_shell_hook_pre_commit(self):
    if os.name != 'posix':
        self.skipTest('shell hook tests requires POSIX shell')
    pre_commit_fail = '#!/bin/sh\nexit 1\n'
    pre_commit_success = '#!/bin/sh\nexit 0\n'
    repo_dir = os.path.join(self.mkdtemp())
    self.addCleanup(shutil.rmtree, repo_dir)
    r = Repo.init(repo_dir)
    self.addCleanup(r.close)
    pre_commit = os.path.join(r.controldir(), 'hooks', 'pre-commit')
    with open(pre_commit, 'w') as f:
        f.write(pre_commit_fail)
    os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    self.assertRaises(errors.CommitError, r.do_commit, b'failed commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    with open(pre_commit, 'w') as f:
        f.write(pre_commit_success)
    os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    commit_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual([], r[commit_sha].parents)