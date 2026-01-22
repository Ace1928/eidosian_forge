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
class TestSubunitLogDetails(tests.TestCase, SelfTestHelper):
    _test_needs_features = [features.subunit]

    def run_subunit_stream(self, test_name):
        from subunit import ProtocolTestCase

        def factory():
            return TestUtil.TestSuite([_get_test(test_name)])
        stream = self.run_selftest(runner_class=tests.SubUnitBzrRunnerv1, test_suite_factory=factory)
        test = ProtocolTestCase(stream)
        result = testtools.TestResult()
        test.run(result)
        content = stream.getvalue()
        return (content, result)

    def test_fail_has_log(self):
        content, result = self.run_subunit_stream('test_fail')
        self.assertEqual(1, len(result.failures))
        self.assertContainsRe(content, b'(?m)^log$')
        self.assertContainsRe(content, b'this test will fail')

    def test_error_has_log(self):
        content, result = self.run_subunit_stream('test_error')
        self.assertContainsRe(content, b'(?m)^log$')
        self.assertContainsRe(content, b'this test errored')

    def test_skip_has_no_log(self):
        content, result = self.run_subunit_stream('test_skip')
        self.assertNotContainsRe(content, b'(?m)^log$')
        self.assertNotContainsRe(content, b'this test will be skipped')
        reasons = result.skip_reasons
        self.assertEqual({'reason'}, set(reasons))
        skips = reasons['reason']
        self.assertEqual(1, len(skips))

    def test_missing_feature_has_no_log(self):
        content, result = self.run_subunit_stream('test_missing_feature')
        self.assertNotContainsRe(content, b'(?m)^log$')
        self.assertNotContainsRe(content, b'missing the feature')
        reasons = result.skip_reasons
        self.assertEqual({'_MissingFeature\n'}, set(reasons))
        skips = reasons['_MissingFeature\n']
        self.assertEqual(1, len(skips))

    def test_xfail_has_no_log(self):
        content, result = self.run_subunit_stream('test_xfail')
        self.assertNotContainsRe(content, b'(?m)^log$')
        self.assertNotContainsRe(content, b'test with expected failure')
        self.assertEqual(1, len(result.expectedFailures))
        result_content = result.expectedFailures[0][1]
        self.assertNotContainsRe(result_content, '(?m)^(?:Text attachment: )?log(?:$|: )')
        self.assertNotContainsRe(result_content, 'test with expected failure')

    def test_unexpected_success_has_log(self):
        content, result = self.run_subunit_stream('test_unexpected_success')
        self.assertContainsRe(content, b'(?m)^log$')
        self.assertContainsRe(content, b'test with unexpected success')
        self.assertEqual(1, len(result.unexpectedSuccesses))

    def test_success_has_no_log(self):
        content, result = self.run_subunit_stream('test_success')
        self.assertEqual(1, result.testsRun)
        self.assertNotContainsRe(content, b'(?m)^log$')
        self.assertNotContainsRe(content, b'this test succeeds')