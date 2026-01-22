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
class CreateRepositoryTests(TestCase):

    def assertFileContentsEqual(self, expected, repo, path):
        f = repo.get_named_file(path)
        if not f:
            self.assertEqual(expected, None)
        else:
            with f:
                self.assertEqual(expected, f.read())

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

    def test_create_memory(self):
        repo = MemoryRepo.init_bare([], {})
        self._check_repo_contents(repo, True)

    def test_create_disk_bare(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init_bare(tmp_dir)
        self.assertEqual(tmp_dir, repo._controldir)
        self._check_repo_contents(repo, True)

    def test_create_disk_non_bare(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(tmp_dir)
        self.assertEqual(os.path.join(tmp_dir, '.git'), repo._controldir)
        self._check_repo_contents(repo, False)

    def test_create_disk_non_bare_mkdir(self):
        tmp_dir = tempfile.mkdtemp()
        target_dir = os.path.join(tmp_dir, 'target')
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(target_dir, mkdir=True)
        self.assertEqual(os.path.join(target_dir, '.git'), repo._controldir)
        self._check_repo_contents(repo, False)

    def test_create_disk_bare_mkdir(self):
        tmp_dir = tempfile.mkdtemp()
        target_dir = os.path.join(tmp_dir, 'target')
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init_bare(target_dir, mkdir=True)
        self.assertEqual(target_dir, repo._controldir)
        self._check_repo_contents(repo, True)