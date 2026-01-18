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
def test_shell_hook_post_commit(self):
    if os.name != 'posix':
        self.skipTest('shell hook tests requires POSIX shell')
    repo_dir = self.mkdtemp()
    self.addCleanup(shutil.rmtree, repo_dir)
    r = Repo.init(repo_dir)
    self.addCleanup(r.close)
    fd, path = tempfile.mkstemp(dir=repo_dir)
    os.close(fd)
    post_commit_msg = '#!/bin/sh\nrm ' + path + '\n'
    root_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    self.assertEqual([], r[root_sha].parents)
    post_commit = os.path.join(r.controldir(), 'hooks', 'post-commit')
    with open(post_commit, 'wb') as f:
        f.write(post_commit_msg.encode(locale.getpreferredencoding()))
    os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    commit_sha = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    self.assertEqual([root_sha], r[commit_sha].parents)
    self.assertFalse(os.path.exists(path))
    post_commit_msg_fail = '#!/bin/sh\nexit 1\n'
    with open(post_commit, 'w') as f:
        f.write(post_commit_msg_fail)
    os.chmod(post_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    warnings.simplefilter('always', UserWarning)
    self.addCleanup(warnings.resetwarnings)
    warnings_list, restore_warnings = setup_warning_catcher()
    self.addCleanup(restore_warnings)
    commit_sha2 = r.do_commit(b'empty commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12345, commit_timezone=0, author_timestamp=12345, author_timezone=0)
    expected_warning = UserWarning('post-commit hook failed: Hook post-commit exited with non-zero status 1')
    for w in warnings_list:
        if type(w) is type(expected_warning) and w.args == expected_warning.args:
            break
    else:
        raise AssertionError(f'Expected warning {expected_warning!r} not in {warnings_list!r}')
    self.assertEqual([commit_sha], r[commit_sha2].parents)