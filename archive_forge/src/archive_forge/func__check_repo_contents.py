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
def _check_repo_contents(self, repo, expect_bare):
    self.assertEqual(expect_bare, repo.bare)
    self.assertFileContentsEqual(b'Unnamed repository', repo, 'description')
    self.assertFileContentsEqual(b'', repo, os.path.join('info', 'exclude'))
    self.assertFileContentsEqual(None, repo, 'nonexistent file')
    barestr = b'bare = ' + str(expect_bare).lower().encode('ascii')
    with repo.get_named_file('config') as f:
        config_text = f.read()
        self.assertIn(barestr, config_text, '%r' % config_text)
    expect_filemode = sys.platform != 'win32'
    barestr = b'filemode = ' + str(expect_filemode).lower().encode('ascii')
    with repo.get_named_file('config') as f:
        config_text = f.read()
        self.assertIn(barestr, config_text, '%r' % config_text)
    if isinstance(repo, Repo):
        expected_mode = '0o100644' if expect_filemode else '0o100666'
        expected = {'HEAD': expected_mode, 'config': expected_mode, 'description': expected_mode}
        actual = {f[len(repo._controldir) + 1:]: oct(os.stat(f).st_mode) for f in glob.glob(os.path.join(repo._controldir, '*')) if os.path.isfile(f)}
        self.assertEqual(expected, actual)