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
def test_unsupported_features_listed(self):
    """When unsupported features are encountered they are detailed."""

    class Feature1(features.Feature):

        def _probe(self):
            return False

    class Feature2(features.Feature):

        def _probe(self):
            return False
    test1 = SampleTestCase('_test_pass')
    test1._test_needs_features = [Feature1()]
    test2 = SampleTestCase('_test_pass')
    test2._test_needs_features = [Feature2()]
    test = unittest.TestSuite()
    test.addTest(test1)
    test.addTest(test2)
    stream = StringIO()
    runner = tests.TextTestRunner(stream=stream)
    self.run_test_runner(runner, test)
    lines = stream.getvalue().splitlines()
    self.assertEqual(['OK', "Missing feature 'Feature1' skipped 1 tests.", "Missing feature 'Feature2' skipped 1 tests."], lines[-3:])