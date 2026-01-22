import gzip
import os
import time
from io import BytesIO
from dulwich import porcelain
from dulwich.errors import HangupException
from dulwich.repo import Repo as GitRepo
from ...branch import Branch
from ...controldir import BranchReferenceLoop, ControlDir
from ...errors import (ConnectionReset, DivergedBranches, NoSuchTag,
from ...tests import TestCase, TestCaseWithTransport
from ...tests.features import ExecutableFeature
from ...urlutils import join as urljoin
from ..mapping import default_mapping
from ..remote import (GitRemoteRevisionTree, GitSmartRemoteNotSupported,
from ..tree import MissingNestedTree
class SplitUrlTests(TestCase):

    def test_simple(self):
        self.assertEqual(('foo', None, None, '/bar'), split_git_url('git://foo/bar'))

    def test_port(self):
        self.assertEqual(('foo', 343, None, '/bar'), split_git_url('git://foo:343/bar'))

    def test_username(self):
        self.assertEqual(('foo', None, 'la', '/bar'), split_git_url('git://la@foo/bar'))

    def test_username_password(self):
        self.assertEqual(('foo', None, 'la', '/bar'), split_git_url('git://la:passwd@foo/bar'))

    def test_nopath(self):
        self.assertEqual(('foo', None, None, '/'), split_git_url('git://foo/'))

    def test_slashpath(self):
        self.assertEqual(('foo', None, None, '//bar'), split_git_url('git://foo//bar'))

    def test_homedir(self):
        self.assertEqual(('foo', None, None, '~bar'), split_git_url('git://foo/~bar'))

    def test_file(self):
        self.assertEqual(('', None, None, '/bar'), split_git_url('file:///bar'))