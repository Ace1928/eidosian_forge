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
class DescribeTests(PorcelainTestCase):

    def test_no_commits(self):
        self.assertRaises(KeyError, porcelain.describe, self.repo.path)

    def test_single_commit(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('BAR')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        sha = porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        self.assertEqual('g{}'.format(sha[:7].decode('ascii')), porcelain.describe(self.repo.path))

    def test_tag(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('BAR')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        porcelain.tag_create(self.repo.path, b'tryme', b'foo <foo@bar.com>', b'bar', annotated=True)
        self.assertEqual('tryme', porcelain.describe(self.repo.path))

    def test_tag_and_commit(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('BAR')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        porcelain.tag_create(self.repo.path, b'tryme', b'foo <foo@bar.com>', b'bar', annotated=True)
        with open(fullpath, 'w') as f:
            f.write('BAR2')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        sha = porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        self.assertEqual('tryme-1-g{}'.format(sha[:7].decode('ascii')), porcelain.describe(self.repo.path))

    def test_tag_and_commit_full(self):
        fullpath = os.path.join(self.repo.path, 'foo')
        with open(fullpath, 'w') as f:
            f.write('BAR')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        porcelain.tag_create(self.repo.path, b'tryme', b'foo <foo@bar.com>', b'bar', annotated=True)
        with open(fullpath, 'w') as f:
            f.write('BAR2')
        porcelain.add(repo=self.repo.path, paths=[fullpath])
        sha = porcelain.commit(self.repo.path, message=b'Some message', author=b'Joe <joe@example.com>', committer=b'Bob <bob@example.com>')
        self.assertEqual('tryme-1-g{}'.format(sha.decode('ascii')), porcelain.describe(self.repo.path, abbrev=40))