import os
import re
import shutil
import tempfile
from io import BytesIO
from dulwich.tests import TestCase
from ..ignore import (
from ..repo import Repo
class IgnoreFilterManagerTests(TestCase):

    def test_load_ignore(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(tmp_dir)
        with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
            f.write(b'/foo/bar\n')
            f.write(b'/dir2\n')
            f.write(b'/dir3/\n')
        os.mkdir(os.path.join(repo.path, 'dir'))
        with open(os.path.join(repo.path, 'dir', '.gitignore'), 'wb') as f:
            f.write(b'/blie\n')
        with open(os.path.join(repo.path, 'dir', 'blie'), 'wb') as f:
            f.write(b'IGNORED')
        p = os.path.join(repo.controldir(), 'info', 'exclude')
        with open(p, 'wb') as f:
            f.write(b'/excluded\n')
        m = IgnoreFilterManager.from_repo(repo)
        self.assertTrue(m.is_ignored('dir/blie'))
        self.assertIs(None, m.is_ignored(os.path.join('dir', 'bloe')))
        self.assertIs(None, m.is_ignored('dir'))
        self.assertTrue(m.is_ignored(os.path.join('foo', 'bar')))
        self.assertTrue(m.is_ignored(os.path.join('excluded')))
        self.assertTrue(m.is_ignored(os.path.join('dir2', 'fileinignoreddir')))
        self.assertFalse(m.is_ignored('dir3'))
        self.assertTrue(m.is_ignored('dir3/'))
        self.assertTrue(m.is_ignored('dir3/bla'))

    def test_nested_gitignores(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(tmp_dir)
        with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
            f.write(b'/*\n')
            f.write(b'!/foo\n')
        os.mkdir(os.path.join(repo.path, 'foo'))
        with open(os.path.join(repo.path, 'foo', '.gitignore'), 'wb') as f:
            f.write(b'/bar\n')
        with open(os.path.join(repo.path, 'foo', 'bar'), 'wb') as f:
            f.write(b'IGNORED')
        m = IgnoreFilterManager.from_repo(repo)
        self.assertTrue(m.is_ignored('foo/bar'))

    def test_load_ignore_ignorecase(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(tmp_dir)
        config = repo.get_config()
        config.set(b'core', b'ignorecase', True)
        config.write_to_path()
        with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
            f.write(b'/foo/bar\n')
            f.write(b'/dir\n')
        m = IgnoreFilterManager.from_repo(repo)
        self.assertTrue(m.is_ignored(os.path.join('dir', 'blie')))
        self.assertTrue(m.is_ignored(os.path.join('DIR', 'blie')))

    def test_ignored_contents(self):
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir)
        repo = Repo.init(tmp_dir)
        with open(os.path.join(repo.path, '.gitignore'), 'wb') as f:
            f.write(b'a/*\n')
            f.write(b'!a/*.txt\n')
        m = IgnoreFilterManager.from_repo(repo)
        os.mkdir(os.path.join(repo.path, 'a'))
        self.assertIs(None, m.is_ignored('a'))
        self.assertIs(None, m.is_ignored('a/'))
        self.assertFalse(m.is_ignored('a/b.txt'))
        self.assertTrue(m.is_ignored('a/c.dat'))