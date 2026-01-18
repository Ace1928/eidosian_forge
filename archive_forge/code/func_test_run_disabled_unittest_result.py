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
def test_run_disabled_unittest_result(self):
    """Test our compatibility for disabled tests with unittest results."""
    test = SampleTestCase('_test_pass')

    class DisabledFeature:

        def available(self):
            return False
    test._test_needs_features = [DisabledFeature()]
    result = unittest.TestResult()
    test.run(result)
    self.assertEqual(1, result.testsRun)
    self.assertEqual([], result.errors)
    self.assertEqual([], result.failures)