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
class TestTestCaseLogDetails(tests.TestCase):

    def _run_test(self, test_name):
        test = _get_test(test_name)
        result = testtools.TestResult()
        test.run(result)
        return result

    def test_fail_has_log(self):
        result = self._run_test('test_fail')
        self.assertEqual(1, len(result.failures))
        result_content = result.failures[0][1]
        self.assertContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertContainsRe(result_content, 'this was a failing test')

    def test_error_has_log(self):
        result = self._run_test('test_error')
        self.assertEqual(1, len(result.errors))
        result_content = result.errors[0][1]
        self.assertContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertContainsRe(result_content, 'this test errored')

    def test_skip_has_no_log(self):
        result = self._run_test('test_skip')
        reasons = result.skip_reasons
        self.assertEqual({'reason'}, set(reasons))
        skips = reasons['reason']
        self.assertEqual(1, len(skips))
        test = skips[0]
        self.assertFalse('log' in test.getDetails())

    def test_missing_feature_has_no_log(self):
        result = self._run_test('test_missing_feature')
        reasons = result.skip_reasons
        self.assertEqual({str(missing_feature)}, set(reasons))
        skips = reasons[str(missing_feature)]
        self.assertEqual(1, len(skips))
        test = skips[0]
        self.assertFalse('log' in test.getDetails())

    def test_xfail_has_no_log(self):
        result = self._run_test('test_xfail')
        self.assertEqual(1, len(result.expectedFailures))
        result_content = result.expectedFailures[0][1]
        self.assertNotContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertNotContainsRe(result_content, 'test with expected failure')

    def test_unexpected_success_has_log(self):
        result = self._run_test('test_unexpected_success')
        self.assertEqual(1, len(result.unexpectedSuccesses))
        test = result.unexpectedSuccesses[0]
        details = test.getDetails()
        self.assertTrue('log' in details)