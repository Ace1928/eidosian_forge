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
def test_verbose_test_count(self):
    """A verbose test run reports the right test count at the start"""
    suite = TestUtil.TestSuite([unittest.FunctionTestCase(lambda: None), unittest.FunctionTestCase(lambda: None)])
    self.assertEqual(suite.countTestCases(), 2)
    stream = StringIO()
    runner = tests.TextTestRunner(stream=stream, verbosity=2)
    self.run_test_runner(runner, tests.CountingDecorator(suite))
    self.assertStartsWith(stream.getvalue(), 'running 2 tests')