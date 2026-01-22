from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import functools
import os
import signal
import six
import threading
import textwrap
import time
from unittest import mock
import boto
from boto.storage_uri import BucketStorageUri
from boto.storage_uri import StorageUri
from gslib import cs_api_map
from gslib import command
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import DummyArgChecker
from gslib.tests.mock_cloud_api import MockCloudApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.base import RequiresIsolation
from gslib.tests.util import unittest
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.system_util import IS_OSX
from gslib.utils.system_util import IS_WINDOWS
class TestParallelismFramework(testcase.GsUtilUnitTestCase):
    """gsutil parallelism framework test suite."""
    command_class = FakeCommand

    def _RunApply(self, func, args_iterator, process_count, thread_count, command_inst=None, shared_attrs=None, fail_on_error=False, thr_exc_handler=None, arg_checker=DummyArgChecker):
        command_inst = command_inst or self.command_class(True)
        exception_handler = thr_exc_handler or _ExceptionHandler
        return command_inst.Apply(func, args_iterator, exception_handler, thread_count=thread_count, process_count=process_count, arg_checker=arg_checker, should_return_results=True, shared_attrs=shared_attrs, fail_on_error=fail_on_error)

    @RequiresIsolation
    def testBasicApplySingleProcessSingleThread(self):
        self._TestBasicApply(1, 1)

    @RequiresIsolation
    def testBasicApplySingleProcessMultiThread(self):
        self._TestBasicApply(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testBasicApplyMultiProcessSingleThread(self):
        self._TestBasicApply(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testBasicApplyMultiProcessMultiThread(self):
        self._TestBasicApply(3, 3)

    @Timeout
    def _TestBasicApply(self, process_count, thread_count):
        args = [()] * (17 * process_count * thread_count + 1)
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(len(args), len(results))

    @RequiresIsolation
    def testNoTasksSingleProcessSingleThread(self):
        self._TestApplyWithNoTasks(1, 1)

    @RequiresIsolation
    def testNoTasksSingleProcessMultiThread(self):
        self._TestApplyWithNoTasks(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testNoTasksMultiProcessSingleThread(self):
        self._TestApplyWithNoTasks(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testNoTasksMultiProcessMultiThread(self):
        self._TestApplyWithNoTasks(3, 3)

    @Timeout
    def _TestApplyWithNoTasks(self, process_count, thread_count):
        """Tests that calling Apply with no tasks releases locks/semaphores."""
        empty_args = [()]
        for _ in range(process_count * thread_count + 1):
            self._RunApply(_ReturnOneValue, empty_args, process_count, thread_count)
        self._TestBasicApply(process_count, thread_count)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testApplySaturatesMultiProcessSingleThread(self):
        self._TestApplySaturatesAvailableProcessesAndThreads(3, 1)

    @RequiresIsolation
    def testApplySaturatesSingleProcessMultiThread(self):
        self._TestApplySaturatesAvailableProcessesAndThreads(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testApplySaturatesMultiProcessMultiThread(self):
        self._TestApplySaturatesAvailableProcessesAndThreads(3, 3)

    @RequiresIsolation
    def _TestApplySaturatesAvailableProcessesAndThreads(self, process_count, thread_count):
        """Tests that created processes and threads evenly share tasks."""
        calls_per_thread = 2
        args = [()] * (process_count * thread_count * calls_per_thread)
        expected_calls_per_thread = calls_per_thread
        if not self.command_class(True).multiprocessing_is_available:
            expected_calls_per_thread = calls_per_thread * process_count
        results = self._RunApply(_SleepThenReturnProcAndThreadId, args, process_count, thread_count)
        usage_dict = {}
        for process_id, thread_id in results:
            usage_dict[process_id, thread_id] = usage_dict.get((process_id, thread_id), 0) + 1
        for id_tuple, num_tasks_completed in six.iteritems(usage_dict):
            self.assertEqual(num_tasks_completed, expected_calls_per_thread, 'Process %s thread %s completed %s tasks. Expected: %s' % (id_tuple[0], id_tuple[1], num_tasks_completed, expected_calls_per_thread))

    @RequiresIsolation
    def testIteratorFailureSingleProcessSingleThread(self):
        self._TestIteratorFailure(1, 1)

    @RequiresIsolation
    def testIteratorFailureSingleProcessMultiThread(self):
        self._TestIteratorFailure(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testIteratorFailureMultiProcessSingleThread(self):
        self._TestIteratorFailure(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testIteratorFailureMultiProcessMultiThread(self):
        self._TestIteratorFailure(3, 3)

    @Timeout
    def _TestIteratorFailure(self, process_count, thread_count):
        """Tests apply with a failing iterator."""
        args = FailingIterator(10, [0])
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(9, len(results))
        args = FailingIterator(10, [5])
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(9, len(results))
        args = FailingIterator(10, [9])
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(9, len(results))
        if process_count * thread_count > 1:
            args = FailingIterator(10, [9])
            results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, fail_on_error=True)
            self.assertEqual(9, len(results))
        args = FailingIterator(10, range(10))
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(0, len(results))
        args = FailingIterator(0, [])
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count)
        self.assertEqual(0, len(results))

    @RequiresIsolation
    def testTestSharedAttrsWorkSingleProcessSingleThread(self):
        self._TestSharedAttrsWork(1, 1)

    @RequiresIsolation
    def testTestSharedAttrsWorkSingleProcessMultiThread(self):
        self._TestSharedAttrsWork(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testTestSharedAttrsWorkMultiProcessSingleThread(self):
        self._TestSharedAttrsWork(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testTestSharedAttrsWorkMultiProcessMultiThread(self):
        self._TestSharedAttrsWork(3, 3)

    @Timeout
    def _TestSharedAttrsWork(self, process_count, thread_count):
        """Tests that Apply successfully uses shared_attrs."""
        command_inst = self.command_class(True)
        command_inst.arg_length_sum = 19
        args = ['foo', ['bar', 'baz'], [], ['x', 'y'], [], 'abcd']
        self._RunApply(_IncrementByLength, args, process_count, thread_count, command_inst=command_inst, shared_attrs=['arg_length_sum'])
        expected_sum = 19
        for arg in args:
            expected_sum += len(arg)
        self.assertEqual(expected_sum, command_inst.arg_length_sum)
        for failing_iterator, expected_failure_count in ((FailingIterator(5, [0]), 1), (FailingIterator(10, [1, 3, 5]), 3), (FailingIterator(5, [4]), 1)):
            command_inst = self.command_class(True)
            args = failing_iterator
            self._RunApply(_ReturnOneValue, args, process_count, thread_count, command_inst=command_inst, shared_attrs=['failure_count'])
            self.assertEqual(expected_failure_count, command_inst.failure_count, msg='Failure count did not match. Expected: %s, actual: %s for failing iterator of size %s, failing indices %s' % (expected_failure_count, command_inst.failure_count, failing_iterator.size, failing_iterator.failure_indices))

    @RequiresIsolation
    def testThreadsSurviveExceptionsInFuncSingleProcessSingleThread(self):
        self._TestThreadsSurviveExceptionsInFunc(1, 1)

    @RequiresIsolation
    def testThreadsSurviveExceptionsInFuncSingleProcessMultiThread(self):
        self._TestThreadsSurviveExceptionsInFunc(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testThreadsSurviveExceptionsInFuncMultiProcessSingleThread(self):
        self._TestThreadsSurviveExceptionsInFunc(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testThreadsSurviveExceptionsInFuncMultiProcessMultiThread(self):
        self._TestThreadsSurviveExceptionsInFunc(3, 3)

    @Timeout
    def _TestThreadsSurviveExceptionsInFunc(self, process_count, thread_count):
        command_inst = self.command_class(True)
        args = [()] * 5
        self._RunApply(_FailureFunc, args, process_count, thread_count, command_inst=command_inst, shared_attrs=['failure_count'], thr_exc_handler=_FailingExceptionHandler)
        self.assertEqual(len(args), command_inst.failure_count)

    @RequiresIsolation
    def testThreadsSurviveExceptionsInHandlerSingleProcessSingleThread(self):
        self._TestThreadsSurviveExceptionsInHandler(1, 1)

    @RequiresIsolation
    def testThreadsSurviveExceptionsInHandlerSingleProcessMultiThread(self):
        self._TestThreadsSurviveExceptionsInHandler(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testThreadsSurviveExceptionsInHandlerMultiProcessSingleThread(self):
        self._TestThreadsSurviveExceptionsInHandler(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testThreadsSurviveExceptionsInHandlerMultiProcessMultiThread(self):
        self._TestThreadsSurviveExceptionsInHandler(3, 3)

    @Timeout
    def _TestThreadsSurviveExceptionsInHandler(self, process_count, thread_count):
        command_inst = self.command_class(True)
        args = [()] * 5
        self._RunApply(_FailureFunc, args, process_count, thread_count, command_inst=command_inst, shared_attrs=['failure_count'], thr_exc_handler=_FailingExceptionHandler)
        self.assertEqual(len(args), command_inst.failure_count)

    @RequiresIsolation
    @Timeout
    def testFailOnErrorFlag(self):
        """Tests that fail_on_error produces the correct exception on failure."""

        def _ExpectCustomException(test_func):
            try:
                test_func()
                self.fail('Setting fail_on_error should raise any exception encountered.')
            except CustomException as e:
                pass
            except Exception as e:
                self.fail('Got unexpected error: ' + str(e))

        def _RunFailureFunc():
            command_inst = self.command_class(True)
            args = [()] * 5
            self._RunApply(_FailureFunc, args, 1, 1, command_inst=command_inst, shared_attrs=['failure_count'], fail_on_error=True)
        _ExpectCustomException(_RunFailureFunc)

        def _RunFailingIteratorFirstPosition():
            args = FailingIterator(10, [0])
            results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
            self.assertEqual(0, len(results))
        _ExpectCustomException(_RunFailingIteratorFirstPosition)

        def _RunFailingIteratorPositionMiddlePosition():
            args = FailingIterator(10, [5])
            results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
            self.assertEqual(5, len(results))
        _ExpectCustomException(_RunFailingIteratorPositionMiddlePosition)

        def _RunFailingIteratorLastPosition():
            args = FailingIterator(10, [9])
            results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
            self.assertEqual(9, len(results))
        _ExpectCustomException(_RunFailingIteratorLastPosition)

        def _RunFailingIteratorMultiplePositions():
            args = FailingIterator(10, [1, 3, 5])
            results = self._RunApply(_ReturnOneValue, args, 1, 1, fail_on_error=True)
            self.assertEqual(1, len(results))
        _ExpectCustomException(_RunFailingIteratorMultiplePositions)

    @RequiresIsolation
    def testRecursiveDepthThreeDifferentFunctionsSingleProcessSingleThread(self):
        self._TestRecursiveDepthThreeDifferentFunctions(1, 1)

    @RequiresIsolation
    def testRecursiveDepthThreeDifferentFunctionsSingleProcessMultiThread(self):
        self._TestRecursiveDepthThreeDifferentFunctions(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testRecursiveDepthThreeDifferentFunctionsMultiProcessSingleThread(self):
        self._TestRecursiveDepthThreeDifferentFunctions(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testRecursiveDepthThreeDifferentFunctionsMultiProcessMultiThread(self):
        self._TestRecursiveDepthThreeDifferentFunctions(3, 3)

    @RequiresIsolation
    @unittest.skipUnless(IS_OSX, 'This warning should only be printed on MacOS')
    def testMacOSLogsMultiprocessingWarning(self):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._TestRecursiveDepthThreeDifferentFunctions(3, 3)
        macos_message = 'If you experience problems with multiprocessing on MacOS'
        contains_message = [message.startswith(macos_message) for message in mock_log_handler.messages['info']]
        self.assertTrue(any(contains_message))
        logger.removeHandler(mock_log_handler)

    @RequiresIsolation
    @unittest.skipIf(IS_OSX, 'This warning should be printed on MacOS')
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testNonMacOSDoesNotLogMultiprocessingWarning(self):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._TestRecursiveDepthThreeDifferentFunctions(3, 3)
        macos_message = 'If you experience problems with multiprocessing on MacOS'
        contains_message = [message.startswith(macos_message) for message in mock_log_handler.messages['info']]
        self.assertFalse(any(contains_message))
        logger.removeHandler(mock_log_handler)

    @RequiresIsolation
    def testMultithreadingDoesNotLogMacOSWarning(self):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._TestRecursiveDepthThreeDifferentFunctions(1, 3)
        macos_message = 'If you experience problems with multiprocessing on MacOS'
        contains_message = [message.startswith(macos_message) for message in mock_log_handler.messages['info']]
        self.assertFalse(any(contains_message))
        logger.removeHandler(mock_log_handler)

    @Timeout
    def _TestRecursiveDepthThreeDifferentFunctions(self, process_count, thread_count):
        """Tests recursive application of Apply.

    Calls Apply(A), where A calls Apply(B) followed by Apply(C) and B calls
    Apply(C).

    Args:
      process_count: Number of processes to use.
      thread_count: Number of threads to use.
    """
        base_args = [3, 1, 4, 1, 5]
        args = [[process_count, thread_count, count] for count in base_args]
        results = self._RunApply(_ReApplyWithReplicatedArguments, args, process_count, thread_count)
        self.assertEqual(7 * (sum(base_args) + len(base_args)), sum(results))

    @RequiresIsolation
    def testExceptionInProducerRaisesAndTerminatesSingleProcessSingleThread(self):
        self._TestExceptionInProducerRaisesAndTerminates(1, 1)

    @RequiresIsolation
    def testExceptionInProducerRaisesAndTerminatesSingleProcessMultiThread(self):
        self._TestExceptionInProducerRaisesAndTerminates(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testExceptionInProducerRaisesAndTerminatesMultiProcessSingleThread(self):
        self._TestExceptionInProducerRaisesAndTerminates(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testExceptionInProducerRaisesAndTerminatesMultiProcessMultiThread(self):
        self._TestExceptionInProducerRaisesAndTerminates(3, 3)

    @Timeout
    def _TestExceptionInProducerRaisesAndTerminates(self, process_count, thread_count):
        args = self
        try:
            self._RunApply(_ReturnOneValue, args, process_count, thread_count)
            self.fail('Did not raise expected exception.')
        except TypeError:
            pass

    @RequiresIsolation
    def testSkippedArgumentsSingleThreadSingleProcess(self):
        self._TestSkippedArguments(1, 1)

    @RequiresIsolation
    def testSkippedArgumentsMultiThreadSingleProcess(self):
        self._TestSkippedArguments(1, 3)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testSkippedArgumentsSingleThreadMultiProcess(self):
        self._TestSkippedArguments(3, 1)

    @RequiresIsolation
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testSkippedArgumentsMultiThreadMultiProcess(self):
        self._TestSkippedArguments(3, 3)

    @Timeout
    def _TestSkippedArguments(self, process_count, thread_count):
        n = 2 * process_count * thread_count
        args = range(1, n + 1)
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, arg_checker=_SkipEvenNumbersArgChecker)
        self.assertEqual(n / 2, len(results))
        self.assertEqual(n / 2, sum(results))
        args = [2 * x for x in args]
        results = self._RunApply(_ReturnOneValue, args, process_count, thread_count, arg_checker=_SkipEvenNumbersArgChecker)
        self.assertEqual(0, len(results))

    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_THRESHOLD', 2)
    @mock.patch.object(command, 'GetTermLines', return_value=100)
    def testSequentialApplyRecommendsParallelismAfterThreshold(self, mock_get_term_lines):
        mock_get_term_lines.return_value = 100
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._RunApply(_ReturnOneValue, range(2), process_count=1, thread_count=1)
        contains_message = [message == PARALLEL_PROCESSING_MESSAGE for message in mock_log_handler.messages['info']]
        self.assertTrue(any(contains_message))
        logger.removeHandler(mock_log_handler)

    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_THRESHOLD', 100)
    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_FREQUENCY', 10)
    @mock.patch.object(command, 'GetTermLines', return_value=100)
    def testSequentialApplyRecommendsParallelismAtSuggestionFrequency(self, mock_get_term_lines):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._RunApply(_ReturnOneValue, range(30), process_count=1, thread_count=1)
        contains_message = [message == PARALLEL_PROCESSING_MESSAGE for message in mock_log_handler.messages['info']]
        self.assertEqual(sum(contains_message), 3)
        logger.removeHandler(mock_log_handler)

    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_THRESHOLD', 100)
    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_FREQUENCY', 10)
    @mock.patch.object(command, 'GetTermLines', return_value=2)
    def testSequentialApplyRecommendsParallelismAtEndIfLastSuggestionIsOutOfView(self, mock_get_term_lines):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._RunApply(_ReturnOneValue, range(22), process_count=1, thread_count=1)
        contains_message = [message == PARALLEL_PROCESSING_MESSAGE for message in mock_log_handler.messages['info']]
        self.assertEqual(sum(contains_message), 3)
        logger.removeHandler(mock_log_handler)

    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_THRESHOLD', 100)
    @mock.patch.object(command, 'OFFER_GSUTIL_M_SUGGESTION_FREQUENCY', 10)
    @mock.patch.object(command, 'GetTermLines', return_value=3)
    def testSequentialApplyDoesNotRecommendParallelismAtEndIfLastSuggestionInView(self, mock_get_term_lines):
        logger = CreateOrGetGsutilLogger('FakeCommand')
        mock_log_handler = MockLoggingHandler()
        logger.addHandler(mock_log_handler)
        self._RunApply(_ReturnOneValue, range(22), process_count=1, thread_count=1)
        contains_message = [message == PARALLEL_PROCESSING_MESSAGE for message in mock_log_handler.messages['info']]
        self.assertEqual(sum(contains_message), 2)
        logger.removeHandler(mock_log_handler)

    def testResetConnectionPoolDeletesConnectionState(self):
        StorageUri.connection = mock.Mock(spec=boto.s3.connection.S3Connection)
        StorageUri.provider_pool = {'s3': mock.Mock(spec=boto.s3.connection.S3Connection)}
        self.command_class(True)._ResetConnectionPool()
        self.assertIsNone(StorageUri.connection)
        self.assertFalse(StorageUri.provider_pool)