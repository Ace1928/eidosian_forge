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
class CheckIgnoreTests(PorcelainTestCase):

    def test_check_ignored(self):
        with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
            f.write('foo')
        foo_path = os.path.join(self.repo.path, 'foo')
        with open(foo_path, 'w') as f:
            f.write('BAR')
        bar_path = os.path.join(self.repo.path, 'bar')
        with open(bar_path, 'w') as f:
            f.write('BAR')
        self.assertEqual(['foo'], list(porcelain.check_ignore(self.repo, [foo_path])))
        self.assertEqual([], list(porcelain.check_ignore(self.repo, [bar_path])))

    def test_check_added_abs(self):
        path = os.path.join(self.repo.path, 'foo')
        with open(path, 'w') as f:
            f.write('BAR')
        self.repo.stage(['foo'])
        with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
            f.write('foo\n')
        self.assertEqual([], list(porcelain.check_ignore(self.repo, [path])))
        self.assertEqual(['foo'], list(porcelain.check_ignore(self.repo, [path], no_index=True)))

    def test_check_added_rel(self):
        with open(os.path.join(self.repo.path, 'foo'), 'w') as f:
            f.write('BAR')
        self.repo.stage(['foo'])
        with open(os.path.join(self.repo.path, '.gitignore'), 'w') as f:
            f.write('foo\n')
        cwd = os.getcwd()
        os.mkdir(os.path.join(self.repo.path, 'bar'))
        os.chdir(os.path.join(self.repo.path, 'bar'))
        try:
            self.assertEqual(list(porcelain.check_ignore(self.repo, ['../foo'])), [])
            self.assertEqual(['../foo'], list(porcelain.check_ignore(self.repo, ['../foo'], no_index=True)))
        finally:
            os.chdir(cwd)