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
def test_shell_hook_pre_commit_add_files(self):
    if os.name != 'posix':
        self.skipTest('shell hook tests requires POSIX shell')
    pre_commit_contents = "#!{executable}\nimport sys\nsys.path.extend({path!r})\nfrom dulwich.repo import Repo\n\nwith open('foo', 'w') as f:\n    f.write('newfile')\n\nr = Repo('.')\nr.stage(['foo'])\n".format(executable=sys.executable, path=[os.path.join(os.path.dirname(__file__), '..', '..'), *sys.path])
    repo_dir = os.path.join(self.mkdtemp())
    self.addCleanup(shutil.rmtree, repo_dir)
    r = Repo.init(repo_dir)
    self.addCleanup(r.close)
    with open(os.path.join(repo_dir, 'blah'), 'w') as f:
        f.write('blah')
    r.stage(['blah'])
    pre_commit = os.path.join(r.controldir(), 'hooks', 'pre-commit')
    with open(pre_commit, 'w') as f:
        f.write(pre_commit_contents)
    os.chmod(pre_commit, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    commit_sha = r.do_commit(b'new commit', committer=b'Test Committer <test@nodomain.com>', author=b'Test Author <test@nodomain.com>', commit_timestamp=12395, commit_timezone=0, author_timestamp=12395, author_timezone=0)
    self.assertEqual([], r[commit_sha].parents)
    tree = r[r[commit_sha].tree]
    self.assertEqual({b'blah', b'foo'}, set(tree))