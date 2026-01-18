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
def test_run_enabled_unittest_result(self):
    """Test we revert to regular behaviour when the test is enabled."""
    test = SampleTestCase('_test_pass')

    class EnabledFeature:

        def available(self):
            return True
    test._test_needs_features = [EnabledFeature()]
    result = unittest.TestResult()
    test.run(result)
    self.assertEqual(1, result.testsRun)
    self.assertEqual([], result.errors)
    self.assertEqual([], result.failures)