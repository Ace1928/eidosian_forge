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
class TestTestCase(tests.TestCase):
    """Tests that test the core breezy TestCase."""

    def test_assertLength_matches_empty(self):
        a_list = []
        self.assertLength(0, a_list)

    def test_assertLength_matches_nonempty(self):
        a_list = [1, 2, 3]
        self.assertLength(3, a_list)

    def test_assertLength_fails_different(self):
        a_list = []
        self.assertRaises(AssertionError, self.assertLength, 1, a_list)

    def test_assertLength_shows_sequence_in_failure(self):
        a_list = [1, 2, 3]
        exception = self.assertRaises(AssertionError, self.assertLength, 2, a_list)
        self.assertEqual('Incorrect length: wanted 2, got 3 for [1, 2, 3]', exception.args[0])

    def test_base_setUp_not_called_causes_failure(self):

        class TestCaseWithBrokenSetUp(tests.TestCase):

            def setUp(self):
                pass

            def test_foo(self):
                pass
        test = TestCaseWithBrokenSetUp('test_foo')
        result = unittest.TestResult()
        test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)

    def test_base_tearDown_not_called_causes_failure(self):

        class TestCaseWithBrokenTearDown(tests.TestCase):

            def tearDown(self):
                pass

            def test_foo(self):
                pass
        test = TestCaseWithBrokenTearDown('test_foo')
        result = unittest.TestResult()
        test.run(result)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(1, result.testsRun)

    def test_debug_flags_sanitised(self):
        """The breezy debug flags should be sanitised by setUp."""
        if 'allow_debug' in tests.selftest_debug_flags:
            raise tests.TestNotApplicable('-Eallow_debug option prevents debug flag sanitisation')
        flags = set()
        if self._lock_check_thorough:
            flags.add('strict_locks')
        self.assertEqual(flags, breezy.debug.debug_flags)

    def change_selftest_debug_flags(self, new_flags):
        self.overrideAttr(tests, 'selftest_debug_flags', set(new_flags))

    def test_allow_debug_flag(self):
        """The -Eallow_debug flag prevents breezy.debug.debug_flags from being
        sanitised (i.e. cleared) before running a test.
        """
        self.change_selftest_debug_flags({'allow_debug'})
        breezy.debug.debug_flags = {'a-flag'}

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                self.flags = set(breezy.debug.debug_flags)
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        flags = {'a-flag'}
        if 'disable_lock_checks' not in tests.selftest_debug_flags:
            flags.add('strict_locks')
        self.assertEqual(flags, self.flags)

    def test_disable_lock_checks(self):
        """The -Edisable_lock_checks flag disables thorough checks."""

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                self.flags = set(breezy.debug.debug_flags)
                self.test_lock_check_thorough = nested_self._lock_check_thorough
        self.change_selftest_debug_flags(set())
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertTrue(self.test_lock_check_thorough)
        self.assertEqual({'strict_locks'}, self.flags)
        self.change_selftest_debug_flags({'disable_lock_checks'})
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertFalse(self.test_lock_check_thorough)
        self.assertEqual(set(), self.flags)

    def test_this_fails_strict_lock_check(self):

        class TestThatRecordsFlags(tests.TestCase):

            def test_foo(nested_self):
                self.flags1 = set(breezy.debug.debug_flags)
                self.thisFailsStrictLockCheck()
                self.flags2 = set(breezy.debug.debug_flags)
        self.change_selftest_debug_flags(set())
        test = TestThatRecordsFlags('test_foo')
        test.run(self.make_test_result())
        self.assertEqual({'strict_locks'}, self.flags1)
        self.assertEqual(set(), self.flags2)

    def test_debug_flags_restored(self):
        """The breezy debug flags should be restored to their original state
        after the test was run, even if allow_debug is set.
        """
        self.change_selftest_debug_flags({'allow_debug'})
        breezy.debug.debug_flags = {'original-state'}

        class TestThatModifiesFlags(tests.TestCase):

            def test_foo(self):
                breezy.debug.debug_flags = {'modified'}
        test = TestThatModifiesFlags('test_foo')
        test.run(self.make_test_result())
        self.assertEqual({'original-state'}, breezy.debug.debug_flags)

    def make_test_result(self):
        """Get a test result that writes to a StringIO."""
        return tests.TextTestResult(StringIO(), descriptions=0, verbosity=1)

    def inner_test(self):
        note('inner_test')

    def outer_child(self):
        note('outer_start')
        self.inner_test = TestTestCase('inner_child')
        result = self.make_test_result()
        self.inner_test.run(result)
        note('outer finish')
        self.addCleanup(osutils.delete_any, self._log_file_name)

    def test_trace_nesting(self):
        original_trace = breezy.trace._trace_file
        outer_test = TestTestCase('outer_child')
        result = self.make_test_result()
        outer_test.run(result)
        self.assertEqual(original_trace, breezy.trace._trace_file)

    def method_that_times_a_bit_twice(self):
        self.time(time.sleep, 0.007)
        self.time(time.sleep, 0.007)

    def test_time_creates_benchmark_in_result(self):
        """The TestCase.time() method accumulates a benchmark time."""
        sample_test = TestTestCase('method_that_times_a_bit_twice')
        output_stream = StringIO()
        result = breezy.tests.VerboseTestResult(output_stream, descriptions=0, verbosity=2)
        sample_test.run(result)
        self.assertContainsRe(output_stream.getvalue(), '\\d+ms\\*\\n$')

    def test_hooks_sanitised(self):
        """The breezy hooks should be sanitised by setUp."""
        self.assertEqual(breezy.branch.BranchHooks(), breezy.branch.Branch.hooks)
        self.assertEqual(breezy.bzr.smart.server.SmartServerHooks(), breezy.bzr.smart.server.SmartTCPServer.hooks)
        self.assertEqual(breezy.commands.CommandHooks(), breezy.commands.Command.hooks)

    def test__gather_lsprof_in_benchmarks(self):
        """When _gather_lsprof_in_benchmarks is on, accumulate profile data.

        Each self.time() call is individually and separately profiled.
        """
        self.requireFeature(features.lsprof_feature)
        self._gather_lsprof_in_benchmarks = True
        self.time(time.sleep, 0.0)
        self.time(time.sleep, 0.003)
        self.assertEqual(2, len(self._benchcalls))
        self.assertEqual((time.sleep, (0.0,), {}), self._benchcalls[0][0])
        self.assertEqual((time.sleep, (0.003,), {}), self._benchcalls[1][0])
        self.assertIsInstance(self._benchcalls[0][1], breezy.lsprof.Stats)
        self.assertIsInstance(self._benchcalls[1][1], breezy.lsprof.Stats)
        del self._benchcalls[:]

    def test_knownFailure(self):
        """Self.knownFailure() should raise a KnownFailure exception."""
        self.assertRaises(tests.KnownFailure, self.knownFailure, 'A Failure')

    def test_open_bzrdir_safe_roots(self):
        transport_server = memory.MemoryServer()
        transport_server.start_server()
        self.addCleanup(transport_server.stop_server)
        t = transport.get_transport_from_url(transport_server.get_url())
        controldir.ControlDir.create(t.base)
        self.assertRaises(errors.BzrError, controldir.ControlDir.open_from_transport, t)
        self.permit_url(t.base)
        self._bzr_selftest_roots.append(t.base)
        controldir.ControlDir.open_from_transport(t)

    def test_requireFeature_available(self):
        """self.requireFeature(available) is a no-op."""

        class Available(features.Feature):

            def _probe(self):
                return True
        feature = Available()
        self.requireFeature(feature)

    def test_requireFeature_unavailable(self):
        """self.requireFeature(unavailable) raises UnavailableFeature."""

        class Unavailable(features.Feature):

            def _probe(self):
                return False
        feature = Unavailable()
        self.assertRaises(tests.UnavailableFeature, self.requireFeature, feature)

    def test_run_no_parameters(self):
        test = SampleTestCase('_test_pass')
        test.run()

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

    def test_run_disabled_supporting_result(self):
        """Test disabled tests behaviour with support aware results."""
        test = SampleTestCase('_test_pass')

        class DisabledFeature:

            def __eq__(self, other):
                return isinstance(other, DisabledFeature)

            def available(self):
                return False
        the_feature = DisabledFeature()
        test._test_needs_features = [the_feature]

        class InstrumentedTestResult(unittest.TestResult):

            def __init__(self):
                unittest.TestResult.__init__(self)
                self.calls = []

            def startTest(self, test):
                self.calls.append(('startTest', test))

            def stopTest(self, test):
                self.calls.append(('stopTest', test))

            def addNotSupported(self, test, feature):
                self.calls.append(('addNotSupported', test, feature))
        result = InstrumentedTestResult()
        test.run(result)
        case = result.calls[0][1]
        self.assertEqual([('startTest', case), ('addNotSupported', case, the_feature), ('stopTest', case)], result.calls)

    def test_start_server_registers_url(self):
        transport_server = memory.MemoryServer()
        self.assertEqual([], self._bzr_selftest_roots)
        self.start_server(transport_server)
        self.assertSubset([transport_server.get_url()], self._bzr_selftest_roots)

    def test_assert_list_raises_on_generator(self):

        def generator_which_will_raise():
            yield 1
            raise _TestException()
        e = self.assertListRaises(_TestException, generator_which_will_raise)
        self.assertIsInstance(e, _TestException)
        e = self.assertListRaises(Exception, generator_which_will_raise)
        self.assertIsInstance(e, _TestException)

    def test_assert_list_raises_on_plain(self):

        def plain_exception():
            raise _TestException()
            return []
        e = self.assertListRaises(_TestException, plain_exception)
        self.assertIsInstance(e, _TestException)
        e = self.assertListRaises(Exception, plain_exception)
        self.assertIsInstance(e, _TestException)

    def test_assert_list_raises_assert_wrong_exception(self):

        class _NotTestException(Exception):
            pass

        def wrong_exception():
            raise _NotTestException()

        def wrong_exception_generator():
            yield 1
            yield 2
            raise _NotTestException()
        self.assertRaises(_NotTestException, self.assertListRaises, _TestException, wrong_exception)
        self.assertRaises(_NotTestException, self.assertListRaises, _TestException, wrong_exception_generator)

    def test_assert_list_raises_no_exception(self):

        def success():
            return []

        def success_generator():
            yield 1
            yield 2
        self.assertRaises(AssertionError, self.assertListRaises, _TestException, success)
        self.assertRaises(AssertionError, self.assertListRaises, _TestException, success_generator)

    def _run_successful_test(self, test):
        result = testtools.TestResult()
        test.run(result)
        self.assertTrue(result.wasSuccessful())
        return result

    def test_overrideAttr_without_value(self):
        self.test_attr = 'original'
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                super().setUp()
                self.orig = self.overrideAttr(obj, 'test_attr')

            def test_value(self):
                self.assertEqual('original', self.orig)
                self.assertEqual('original', obj.test_attr)
                obj.test_attr = 'modified'
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertEqual('original', obj.test_attr)

    def test_overrideAttr_with_value(self):
        self.test_attr = 'original'
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                super().setUp()
                self.orig = self.overrideAttr(obj, 'test_attr', new='modified')

            def test_value(self):
                self.assertEqual('original', self.orig)
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertEqual('original', obj.test_attr)

    def test_overrideAttr_with_no_existing_value_and_value(self):
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                tests.TestCase.setUp(self)
                self.orig = self.overrideAttr(obj, 'test_attr', new='modified')

            def test_value(self):
                self.assertEqual(tests._unitialized_attr, self.orig)
                self.assertEqual('modified', obj.test_attr)
        self._run_successful_test(Test('test_value'))
        self.assertRaises(AttributeError, getattr, obj, 'test_attr')

    def test_overrideAttr_with_no_existing_value_and_no_value(self):
        obj = self

        class Test(tests.TestCase):

            def setUp(self):
                tests.TestCase.setUp(self)
                self.orig = self.overrideAttr(obj, 'test_attr')

            def test_value(self):
                self.assertEqual(tests._unitialized_attr, self.orig)
                self.assertRaises(AttributeError, getattr, obj, 'test_attr')
        self._run_successful_test(Test('test_value'))
        self.assertRaises(AttributeError, getattr, obj, 'test_attr')

    def test_recordCalls(self):
        from breezy.tests import test_selftest
        calls = self.recordCalls(test_selftest, '_add_numbers')
        self.assertEqual(test_selftest._add_numbers(2, 10), 12)
        self.assertEqual(calls, [((2, 10), {})])