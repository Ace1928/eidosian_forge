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
class AddTests(PorcelainTestCase):

    def test_add_default_paths(self):
        fullpath = os.path.join(self.repo.path, 'blah')
        with open(fullpath, 'w') as f:
            f.write('\n')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        porcelain.commit(repo=self.repo.path, message=b'test', author=b'test <email>', committer=b'test <email>')
        with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
            f.write('\n')
        os.mkdir(os.path.join(self.repo.path, 'adir'))
        with open(os.path.join(self.repo.path, 'adir', 'afile'), 'w') as f:
            f.write('\n')
        cwd = os.getcwd()
        try:
            os.chdir(self.repo.path)
            self.assertEqual({'foo', 'blah', 'adir', '.git'}, set(os.listdir('.')))
            self.assertEqual((['foo', os.path.join('adir', 'afile')], set()), porcelain.add(self.repo.path))
        finally:
            os.chdir(cwd)
        index = self.repo.open_index()
        self.assertEqual(sorted(index), [b'adir/afile', b'blah', b'foo'])

    def test_add_default_paths_subdir(self):
        os.mkdir(os.path.join(self.repo.path, 'foo'))
        with open(os.path.join(self.repo.path, 'blah'), 'w') as f:
            f.write('\n')
        with open(os.path.join(self.repo.path, 'foo', 'blie'), 'w') as f:
            f.write('\n')
        cwd = os.getcwd()
        try:
            os.chdir(os.path.join(self.repo.path, 'foo'))
            porcelain.add(repo=self.repo.path)
            porcelain.commit(repo=self.repo.path, message=b'test', author=b'test <email>', committer=b'test <email>')
        finally:
            os.chdir(cwd)
        index = self.repo.open_index()
        self.assertEqual(sorted(index), [b'foo/blie'])

    def test_add_file(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('BAR')
        porcelain.add(self.repo.path, paths=[fullpath])
        self.assertIn(b'foo', self.repo.open_index())

    def test_add_ignored(self):
        with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
            f.write('foo\nsubdir/')
        with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
            f.write('BAR')
        with open(os.path.join(self.repo.path, 'bar'), 'w') as f:
            f.write('BAR')
        os.mkdir(os.path.join(self.repo.path, 'subdir'))
        with open(os.path.join(self.repo.path, 'subdir', 'baz'), 'w') as f:
            f.write('BAZ')
        added, ignored = porcelain.add(self.repo.path, paths=[os.path.join(self.repo.path, 'foo'), os.path.join(self.repo.path, 'bar'), os.path.join(self.repo.path, 'subdir')])
        self.assertIn(b'bar', self.repo.open_index())
        self.assertEqual({'bar'}, set(added))
        self.assertEqual({'foo', os.path.join('subdir', '')}, ignored)

    def test_add_file_absolute_path(self):
        with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
            f.write('BAR')
        porcelain.add(self.repo, paths=[os.path.join(self.repo.path, 'foo')])
        self.assertIn(b'foo', self.repo.open_index())

    def test_add_not_in_repo(self):
        with open(os.path.join(self.test_dir, 'foo'), 'w') as f:
            f.write('BAR')
        self.assertRaises(ValueError, porcelain.add, self.repo, paths=[os.path.join(self.test_dir, 'foo')])
        self.assertRaises((ValueError, FileNotFoundError), porcelain.add, self.repo, paths=['../foo'])
        self.assertEqual([], list(self.repo.open_index()))

    def test_add_file_clrf_conversion(self):
        c = self.repo.get_config()
        c.set('core', 'autocrlf', 'input')
        c.write_to_path()
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'wb') as f:
            f.write(b'line1\r\nline2')
        porcelain.add(self.repo.path, paths=[fullpath])
        index = self.repo.open_index()
        self.assertIn(b'foo', index)
        entry = index[b'foo']
        blob = self.repo[entry.sha]
        self.assertEqual(blob.data, b'line1\nline2')