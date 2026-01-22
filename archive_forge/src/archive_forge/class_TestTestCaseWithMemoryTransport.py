import doctest
import gc
import os
import signal
import sys
import threading
import time
import unittest
import warnings
from functools import reduce
from io import BytesIO, StringIO, TextIOWrapper
import testtools.testresult.doubles
from testtools import ExtendedToOriginalDecorator, MultiTestResult
from testtools.content import Content
from testtools.content_type import ContentType
from testtools.matchers import DocTestMatches, Equals
import breezy
from .. import (branchbuilder, controldir, errors, hooks, lockdir, memorytree,
from ..bzr import (bzrdir, groupcompress_repo, remote, workingtree_3,
from ..git import workingtree as git_workingtree
from ..symbol_versioning import (deprecated_function, deprecated_in,
from ..trace import mutter, note
from ..transport import memory
from . import TestUtil, features, test_lsprof, test_server
class TestTestCaseWithMemoryTransport(tests.TestCaseWithMemoryTransport):

    def test_home_is_non_existant_dir_under_root(self):
        """The test_home_dir for TestCaseWithMemoryTransport is missing.

        This is because TestCaseWithMemoryTransport is for tests that do not
        need any disk resources: they should be hooked into breezy in such a
        way that no global settings are being changed by the test (only a
        few tests should need to do that), and having a missing dir as home is
        an effective way to ensure that this is the case.
        """
        self.assertIsSameRealPath(self.TEST_ROOT + '/MemoryTransportMissingHomeDir', self.test_home_dir)
        self.assertIsSameRealPath(self.test_home_dir, os.environ['HOME'])

    def test_cwd_is_TEST_ROOT(self):
        self.assertIsSameRealPath(self.test_dir, self.TEST_ROOT)
        cwd = osutils.getcwd()
        self.assertIsSameRealPath(self.test_dir, cwd)

    def test_BRZ_HOME_and_HOME_are_bytestrings(self):
        """The $BRZ_HOME and $HOME environment variables should not be unicode.

        See https://bugs.launchpad.net/bzr/+bug/464174
        """
        self.assertIsInstance(os.environ['BRZ_HOME'], str)
        self.assertIsInstance(os.environ['HOME'], str)

    def test_make_branch_and_memory_tree(self):
        """In TestCaseWithMemoryTransport we should not make the branch on disk.

        This is hard to comprehensively robustly test, so we settle for making
        a branch and checking no directory was created at its relpath.
        """
        tree = self.make_branch_and_memory_tree('dir')
        self.assertFalse(osutils.lexists('dir'))
        self.assertIsInstance(tree, memorytree.MemoryTree)

    def test_make_branch_and_memory_tree_with_format(self):
        """make_branch_and_memory_tree should accept a format option."""
        format = bzrdir.BzrDirMetaFormat1()
        format.repository_format = repository.format_registry.get_default()
        tree = self.make_branch_and_memory_tree('dir', format=format)
        self.assertFalse(osutils.lexists('dir'))
        self.assertIsInstance(tree, memorytree.MemoryTree)
        self.assertEqual(format.repository_format.__class__, tree.branch.repository._format.__class__)

    def test_make_branch_builder(self):
        builder = self.make_branch_builder('dir')
        self.assertIsInstance(builder, branchbuilder.BranchBuilder)
        self.assertFalse(osutils.lexists('dir'))

    def test_make_branch_builder_with_format(self):
        format = bzrdir.BzrDirMetaFormat1()
        repo_format = repository.format_registry.get_default()
        format.repository_format = repo_format
        builder = self.make_branch_builder('dir', format=format)
        the_branch = builder.get_branch()
        self.assertFalse(osutils.lexists('dir'))
        self.assertEqual(format.repository_format.__class__, the_branch.repository._format.__class__)
        self.assertEqual(repo_format.get_format_string(), self.get_transport().get_bytes('dir/.bzr/repository/format'))

    def test_make_branch_builder_with_format_name(self):
        builder = self.make_branch_builder('dir', format='knit')
        the_branch = builder.get_branch()
        self.assertFalse(osutils.lexists('dir'))
        dir_format = controldir.format_registry.make_controldir('knit')
        self.assertEqual(dir_format.repository_format.__class__, the_branch.repository._format.__class__)
        self.assertEqual(b'Bazaar-NG Knit Repository Format 1', self.get_transport().get_bytes('dir/.bzr/repository/format'))

    def test_dangling_locks_cause_failures(self):

        class TestDanglingLock(tests.TestCaseWithMemoryTransport):

            def test_function(self):
                t = self.get_transport_from_path('.')
                l = lockdir.LockDir(t, 'lock')
                l.create()
                l.attempt_lock()
        test = TestDanglingLock('test_function')
        result = test.run()
        total_failures = result.errors + result.failures
        if self._lock_check_thorough:
            self.assertEqual(1, len(total_failures))
        else:
            self.assertEqual(0, len(total_failures))