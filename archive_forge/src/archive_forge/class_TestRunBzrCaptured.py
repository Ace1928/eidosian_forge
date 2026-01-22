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
class TestRunBzrCaptured(tests.TestCaseWithTransport):

    def apply_redirected(self, stdin=None, stdout=None, stderr=None, a_callable=None, *args, **kwargs):
        self.stdin = stdin
        self.factory_stdin = getattr(breezy.ui.ui_factory, 'stdin', None)
        self.factory = breezy.ui.ui_factory
        self.working_dir = osutils.getcwd()
        stdout.write('foo\n')
        stderr.write('bar\n')
        return 0

    def test_stdin(self):
        self.run_bzr(['foo', 'bar'], stdin='gam')
        self.assertEqual('gam', self.stdin.read())
        self.assertTrue(self.stdin is self.factory_stdin)
        self.run_bzr(['foo', 'bar'], stdin='zippy')
        self.assertEqual('zippy', self.stdin.read())
        self.assertTrue(self.stdin is self.factory_stdin)

    def test_ui_factory(self):
        current_factory = breezy.ui.ui_factory
        self.run_bzr(['foo'])
        self.assertFalse(current_factory is self.factory)
        self.assertNotEqual(sys.stdout, self.factory.stdout)
        self.assertNotEqual(sys.stderr, self.factory.stderr)
        self.assertEqual('foo\n', self.factory.stdout.getvalue())
        self.assertEqual('bar\n', self.factory.stderr.getvalue())
        self.assertIsInstance(self.factory, tests.TestUIFactory)

    def test_working_dir(self):
        self.build_tree(['one/', 'two/'])
        cwd = osutils.getcwd()
        self.run_bzr(['foo', 'bar'])
        self.assertEqual(cwd, self.working_dir)
        self.run_bzr(['foo', 'bar'], working_dir=None)
        self.assertEqual(cwd, self.working_dir)
        self.run_bzr(['foo', 'bar'], working_dir='one')
        self.assertNotEqual(cwd, self.working_dir)
        self.assertEndsWith(self.working_dir, 'one')
        self.assertEqual(cwd, osutils.getcwd())
        self.run_bzr(['foo', 'bar'], working_dir='two')
        self.assertNotEqual(cwd, self.working_dir)
        self.assertEndsWith(self.working_dir, 'two')
        self.assertEqual(cwd, osutils.getcwd())