from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import tempfile
import pprint
import six
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from boto.storage_uri import BucketStorageUri
from gslib import metrics
from gslib import VERSION
from gslib.cs_api_map import ApiSelector
import gslib.exception
from gslib.gcs_json_api import GcsJsonApi
from gslib.metrics import MetricsCollector
from gslib.metrics_tuple import Metric
from gslib.tests.mock_logging_handler import MockLoggingHandler
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import HAS_S3_CREDS
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import SetBotoConfigForTest
from gslib.tests.util import SkipForParFile
from gslib.tests.util import unittest
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.utils.constants import START_CALLBACK_PER_BYTES
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.unit_util import ONE_KIB
from gslib.utils.unit_util import ONE_MIB
from six import add_move, MovedModule
from six.moves import mock
@SkipForParFile('Do not try spawning the interpreter nested in the archive.')
class TestMetricsIntegrationTests(testcase.GsUtilIntegrationTestCase):
    """Integration tests for analytics data collection."""

    def setUp(self):
        super(TestMetricsIntegrationTests, self).setUp()
        self.original_collector_instance = MetricsCollector.GetCollector()
        MetricsCollector.StartTestCollector('https://example.com', 'user-agent-007', {'a': 'b', 'c': 'd'})
        self.collector = MetricsCollector.GetCollector()

    def tearDown(self):
        super(TestMetricsIntegrationTests, self).tearDown()
        MetricsCollector.StopTestCollector(original_instance=self.original_collector_instance)

    def _RunGsUtilWithAnalyticsOutput(self, cmd, expected_status=0):
        """Runs the gsutil command to check for metrics log output.

    The env value is set so that the metrics collector in the subprocess will
    use testing parameters and output the metrics collected to the debugging
    log, which lets us check for proper collection in the stderr.

    Args:
      cmd: The command to run, as a list.
      expected_status: The expected return code.

    Returns:
      The string of metrics output.
    """
        stderr = self.RunGsUtil(['-d'] + cmd, return_stderr=True, expected_status=expected_status, env_vars={'GSUTIL_TEST_ANALYTICS': '2'})
        return METRICS_LOG_RE.search(stderr).group()

    def _StartObjectPatch(self, *args, **kwargs):
        """Runs mock.patch.object with the given args, and returns the mock object.

    This starts the patcher, returns the mock object, and registers the patcher
    to stop on test teardown.

    Args:
      *args: The args to pass to mock.patch.object()
      **kwargs: The kwargs to pass to mock.patch.object()

    Returns:
      Mock, The result of starting the patcher.
    """
        patcher = mock.patch.object(*args, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _CheckParameterValue(self, param_name, exp_value, metrics_to_search):
        """Checks for a correct key=value pair in a log output string."""
        self.assertIn('{0}={1}'.format(metrics._GA_LABEL_MAP[param_name], exp_value), metrics_to_search)

    @mock.patch('time.time', new=mock.MagicMock(return_value=0))
    def testMetricsReporting(self):
        """Tests the subprocess creation by Popen in metrics.py."""
        popen_mock = self._StartObjectPatch(subprocess, 'Popen')
        metrics_file = tempfile.NamedTemporaryFile()
        metrics_file.close()
        temp_file_mock = self._StartObjectPatch(tempfile, 'NamedTemporaryFile')
        temp_file_mock.return_value = open(metrics_file.name, 'wb')
        self.collector.ReportMetrics()
        self.assertEqual(0, popen_mock.call_count)
        _LogAllTestMetrics()
        metrics.Shutdown()
        call_list = popen_mock.call_args_list
        self.assertEqual(1, len(call_list))
        args = call_list[0]
        self.assertIn('PYTHONPATH', args[1]['env'])
        missing_paths = set(sys.path) - set(args[1]['env']['PYTHONPATH'].split(os.pathsep))
        self.assertEqual(set(), missing_paths)
        with open(metrics_file.name, 'rb') as metrics_file:
            reported_metrics = pickle.load(metrics_file)
        self.assertEqual(COMMAND_AND_ERROR_TEST_METRICS, set(reported_metrics))

    @mock.patch('time.time', new=mock.MagicMock(return_value=0))
    def testMetricsPosting(self):
        """Tests the metrics posting process as performed in metrics_reporter.py."""
        metrics_file = tempfile.NamedTemporaryFile()
        metrics_file_name = metrics_file.name
        metrics_file.close()

        def MetricsTempFileCleanup(file_path):
            try:
                os.unlink(file_path)
            except OSError:
                pass
        self.addCleanup(MetricsTempFileCleanup, metrics_file_name)

        def CollectMetricAndSetLogLevel(log_level, log_file_path):
            metrics.LogCommandParams(command_name='cmd1', subcommands=['action1'], sub_opts=[('optb', ''), ('opta', '')])
            metrics.LogFatalError(gslib.exception.CommandException('test'))
            self.collector.ReportMetrics(wait_for_report=True, log_level=log_level, log_file_path=log_file_path)
            self.assertEqual([], self.collector._metrics)
        metrics.LogCommandParams(global_opts=[('-y', 'value'), ('-z', ''), ('-x', '')])
        CollectMetricAndSetLogLevel(logging.DEBUG, metrics_file.name)
        with open(metrics_file.name, 'rb') as metrics_log:
            log_text = metrics_log.read()
        if six.PY2:
            expected_response = b"Metric(endpoint=u'https://example.com', method=u'POST', body='{0}&cm2=0&ea=cmd1+action1&ec={1}&el={2}&ev=0', user_agent=u'user-agent-007')".format(GLOBAL_DIMENSIONS_URL_PARAMS, metrics._GA_COMMANDS_CATEGORY, VERSION)
        else:
            expected_response = "Metric(endpoint='https://example.com', method='POST', body='{0}&cm2=0&ea=cmd1+action1&ec={1}&el={2}&ev=0', user_agent='user-agent-007')".format(GLOBAL_DIMENSIONS_URL_PARAMS, metrics._GA_COMMANDS_CATEGORY, VERSION).encode('utf_8')
        self.assertIn(expected_response, log_text)
        self.assertIn(b'RESPONSE: 200', log_text)
        CollectMetricAndSetLogLevel(logging.INFO, metrics_file.name)
        with open(metrics_file.name, 'rb') as metrics_log:
            log_text = metrics_log.read()
        self.assertEqual(log_text, b'')
        CollectMetricAndSetLogLevel(logging.WARN, metrics_file.name)
        with open(metrics_file.name, 'rb') as metrics_log:
            log_text = metrics_log.read()
        self.assertEqual(log_text, b'')

    def testMetricsReportingWithFail(self):
        """Tests that metrics reporting error does not throw an exception."""
        popen_mock = self._StartObjectPatch(subprocess, 'Popen')
        popen_mock.side_effect = OSError()
        self.collector._metrics.append('dummy metric')
        self.collector.ReportMetrics()
        self.assertTrue(popen_mock.called)

    def testCommandCollection(self):
        """Tests the collection of commands."""
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['-m', 'acl', 'set', '-a'], expected_status=1)
        self._CheckParameterValue('Event Category', metrics._GA_COMMANDS_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'acl+set', metrics_list)
        self._CheckParameterValue('Global Options', 'd%2Cm', metrics_list)
        self._CheckParameterValue('Command-Level Options', 'a', metrics_list)
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['ver'])
        self._CheckParameterValue('Event Category', metrics._GA_COMMANDS_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'version', metrics_list)
        self._CheckParameterValue('Command Alias', 'ver', metrics_list)

    def testRetryableErrorMetadataCollection(self):
        """Tests that retryable errors are collected on JSON metadata operations."""
        if self.test_api != ApiSelector.JSON:
            return unittest.skip('Retryable errors are only collected in JSON')
        bucket_uri = self.CreateBucket()
        object_uri = self.CreateObject(bucket_uri=bucket_uri, object_name='foo', contents=b'bar')
        self.collector.ga_params[metrics._GA_LABEL_MAP['Command Name']] = 'rsync'
        gsutil_api = GcsJsonApi(BucketStorageUri, logging.getLogger(), RetryableErrorsQueue(), self.default_provider)
        gsutil_api.api_client.num_retries = 2
        gsutil_api.api_client.max_retry_wait = 1
        key = object_uri.get_key()
        src_obj_metadata = apitools_messages.Object(name=key.name, bucket=key.bucket.name, contentType=key.content_type)
        dst_obj_metadata = apitools_messages.Object(bucket=src_obj_metadata.bucket, name=self.MakeTempName('object'), contentType=src_obj_metadata.contentType)
        with mock.patch.object(http_wrapper, '_MakeRequestNoRetry', side_effect=socket.error()):
            _TryExceptAndPass(gsutil_api.CopyObject, src_obj_metadata, dst_obj_metadata)
        if six.PY2:
            self.assertEqual(self.collector.retryable_errors['SocketError'], 1)
        else:
            self.assertEqual(self.collector.retryable_errors['OSError'], 1)
        with mock.patch.object(http_wrapper, '_MakeRequestNoRetry', side_effect=apitools_exceptions.HttpError('unused', 'unused', 'unused')):
            _TryExceptAndPass(gsutil_api.DeleteObject, bucket_uri.bucket_name, object_uri.object_name)
        self.assertEqual(self.collector.retryable_errors['HttpError'], 1)
        self.assertEqual(self.collector.perf_sum_params.num_retryable_network_errors, 1)
        self.assertEqual(self.collector.perf_sum_params.num_retryable_service_errors, 1)

    def testRetryableErrorMediaCollection(self):
        """Tests that retryable errors are collected on JSON media operations."""
        if self.test_api != ApiSelector.JSON:
            return unittest.skip('Retryable errors are only collected in JSON')
        boto_config_for_test = [('GSUtil', 'resumable_threshold', str(ONE_KIB))]
        bucket_uri = self.CreateBucket()
        halt_size = START_CALLBACK_PER_BYTES * 2
        fpath = self.CreateTempFile(contents=b'a' * halt_size)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(_ResumableUploadRetryHandler(5, apitools_exceptions.BadStatusCodeError, ('unused', 'unused', 'unused'))))
        with SetBotoConfigForTest(boto_config_for_test):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
            self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'BadStatusCodeError', metrics_list)
            self._CheckParameterValue('Retryable Errors', '1', metrics_list)
            self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(_JSONForceHTTPErrorCopyCallbackHandler(5, 404)))
        with SetBotoConfigForTest(boto_config_for_test):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
            self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'ResumableUploadStartOverException', metrics_list)
            self._CheckParameterValue('Retryable Errors', '1', metrics_list)
            self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)
        test_callback_file = self.CreateTempFile(contents=pickle.dumps(_JSONForceHTTPErrorCopyCallbackHandler(5, 404)))
        with SetBotoConfigForTest(boto_config_for_test):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['-m', 'cp', '--testcallbackfile', test_callback_file, fpath, suri(bucket_uri)])
            self._CheckParameterValue('Event Category', metrics._GA_ERRORRETRY_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'ResumableUploadStartOverException', metrics_list)
            self._CheckParameterValue('Retryable Errors', '1', metrics_list)
            self._CheckParameterValue('Num Retryable Service Errors', '1', metrics_list)

    def testFatalErrorCollection(self):
        """Tests that fatal errors are collected."""

        def CheckForCommandException(log_output):
            self._CheckParameterValue('Event Category', metrics._GA_ERRORFATAL_CATEGORY, log_output)
            self._CheckParameterValue('Event Action', 'CommandException', log_output)
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['invalid-command'], expected_status=1)
        CheckForCommandException(metrics_list)
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['mb', '-invalid-option'], expected_status=1)
        CheckForCommandException(metrics_list)
        bucket_uri = self.CreateBucket()
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', suri(bucket_uri), suri(bucket_uri)], expected_status=1)
        CheckForCommandException(metrics_list)

    def _GetAndCheckAllNumberMetrics(self, metrics_to_search, multithread=True):
        """Checks number metrics for PerformanceSummary tests.

    Args:
      metrics_to_search: The string of metrics to search.
      multithread: False if the the metrics were collected in a non-multithread
                   situation.

    Returns:
      (slowest_throughput, fastest_throughput, io_time) as floats.
    """

        def _ExtractNumberMetric(param_name):
            extracted_match = re.search(metrics._GA_LABEL_MAP[param_name] + '=(\\d+\\.?\\d*)&', metrics_to_search)
            if not extracted_match:
                self.fail('Could not find %s (%s) in metrics string %s' % (metrics._GA_LABEL_MAP[param_name], param_name, metrics_to_search))
            return float(extracted_match.group(1))
        if multithread:
            thread_idle_time = _ExtractNumberMetric('Thread Idle Time Percent')
            self.assertGreaterEqual(thread_idle_time, 0)
            self.assertLessEqual(thread_idle_time, 1)
        throughput = _ExtractNumberMetric('Average Overall Throughput')
        self.assertGreater(throughput, 0)
        slowest_throughput = _ExtractNumberMetric('Slowest Thread Throughput')
        fastest_throughput = _ExtractNumberMetric('Fastest Thread Throughput')
        self.assertGreaterEqual(fastest_throughput, slowest_throughput)
        self.assertGreater(slowest_throughput, 0)
        self.assertGreater(fastest_throughput, 0)
        io_time = None
        if IS_LINUX:
            io_time = _ExtractNumberMetric('Disk I/O Time')
            self.assertGreaterEqual(io_time, 0)
        return (slowest_throughput, fastest_throughput, io_time)

    def testPerformanceSummaryFileToFile(self):
        """Tests PerformanceSummary collection in a file-to-file transfer."""
        tmpdir1 = self.CreateTempDir()
        tmpdir2 = self.CreateTempDir()
        file_size = ONE_MIB
        self.CreateTempFile(tmpdir=tmpdir1, contents=b'a' * file_size)
        process_count = 1 if IS_WINDOWS else 6
        with SetBotoConfigForTest([('GSUtil', 'parallel_process_count', str(process_count)), ('GSUtil', 'parallel_thread_count', '7')]):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['-m', 'rsync', tmpdir1, tmpdir2])
            self._CheckParameterValue('Event Category', metrics._GA_PERFSUM_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'FileToFile', metrics_list)
            self._CheckParameterValue('Parallelism Strategy', 'fan', metrics_list)
            self._CheckParameterValue('Source URL Type', 'file', metrics_list)
            self._CheckParameterValue('Num Processes', str(process_count), metrics_list)
            self._CheckParameterValue('Num Threads', '7', metrics_list)
            self._CheckParameterValue('Provider Types', 'file', metrics_list)
            self._CheckParameterValue('Size of Files/Objects Transferred', file_size, metrics_list)
            self._CheckParameterValue('Number of Files/Objects Transferred', 1, metrics_list)
            _, _, io_time = self._GetAndCheckAllNumberMetrics(metrics_list)
            if IS_LINUX:
                self.assertGreaterEqual(io_time, 0)

    @SkipForS3('No slice parallelism support for S3.')
    def testPerformanceSummaryFileToCloud(self):
        """Tests PerformanceSummary collection in a file-to-cloud transfer."""
        bucket_uri = self.CreateBucket()
        tmpdir = self.CreateTempDir()
        file_size = 6
        self.CreateTempFile(tmpdir=tmpdir, contents=b'a' * file_size)
        self.CreateTempFile(tmpdir=tmpdir, contents=b'b' * file_size)
        process_count = 1 if IS_WINDOWS else 2
        with SetBotoConfigForTest([('GSUtil', 'parallel_process_count', str(process_count)), ('GSUtil', 'parallel_thread_count', '3'), ('GSUtil', 'parallel_composite_upload_threshold', '1')]):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['rsync', tmpdir, suri(bucket_uri)])
            self._CheckParameterValue('Event Category', metrics._GA_PERFSUM_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'FileToCloud', metrics_list)
            self._CheckParameterValue('Parallelism Strategy', 'slice', metrics_list)
            self._CheckParameterValue('Num Processes', str(process_count), metrics_list)
            self._CheckParameterValue('Num Threads', '3', metrics_list)
            self._CheckParameterValue('Provider Types', 'file%2C' + bucket_uri.scheme, metrics_list)
            self._CheckParameterValue('Size of Files/Objects Transferred', 2 * file_size, metrics_list)
            self._CheckParameterValue('Number of Files/Objects Transferred', 2, metrics_list)
            _, _, io_time = self._GetAndCheckAllNumberMetrics(metrics_list)
            if IS_LINUX:
                self.assertGreaterEqual(io_time, 0)

    @SkipForS3('No slice parallelism support for S3.')
    def testPerformanceSummaryCloudToFile(self):
        """Tests PerformanceSummary collection in a cloud-to-file transfer."""
        bucket_uri = self.CreateBucket()
        file_size = 6
        object_uri = self.CreateObject(bucket_uri=bucket_uri, contents=b'a' * file_size)
        fpath = self.CreateTempFile()
        process_count = 1 if IS_WINDOWS else 4
        with SetBotoConfigForTest([('GSUtil', 'parallel_process_count', str(process_count)), ('GSUtil', 'parallel_thread_count', '5'), ('GSUtil', 'sliced_object_download_threshold', '1'), ('GSUtil', 'test_assume_fast_crcmod', 'True')]):
            metrics_list = self._RunGsUtilWithAnalyticsOutput(['-m', 'cp', suri(object_uri), fpath])
            self._CheckParameterValue('Event Category', metrics._GA_PERFSUM_CATEGORY, metrics_list)
            self._CheckParameterValue('Event Action', 'CloudToFile', metrics_list)
            self._CheckParameterValue('Parallelism Strategy', 'both', metrics_list)
            self._CheckParameterValue('Num Processes', str(process_count), metrics_list)
            self._CheckParameterValue('Num Threads', '5', metrics_list)
            self._CheckParameterValue('Provider Types', 'file%2C' + bucket_uri.scheme, metrics_list)
            self._CheckParameterValue('Number of Files/Objects Transferred', '1', metrics_list)
            self._CheckParameterValue('Size of Files/Objects Transferred', file_size, metrics_list)
            _, _, io_time = self._GetAndCheckAllNumberMetrics(metrics_list)
            if IS_LINUX:
                self.assertGreaterEqual(io_time, 0)

    def testPerformanceSummaryCloudToCloud(self):
        """Tests PerformanceSummary collection in a cloud-to-cloud transfer."""
        bucket1_uri = self.CreateBucket()
        bucket2_uri = self.CreateBucket()
        file_size = 6
        key_uri = self.CreateObject(bucket_uri=bucket1_uri, contents=b'a' * file_size)
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', '-D', suri(key_uri), suri(bucket2_uri)])
        slowest_throughput, fastest_throughput, _ = self._GetAndCheckAllNumberMetrics(metrics_list, multithread=False)
        self.assertEqual(slowest_throughput, fastest_throughput)
        self._CheckParameterValue('Event Category', metrics._GA_PERFSUM_CATEGORY, metrics_list)
        self._CheckParameterValue('Event Action', 'CloudToCloud%2CDaisyChain', metrics_list)
        self._CheckParameterValue('Parallelism Strategy', 'none', metrics_list)
        self._CheckParameterValue('Source URL Type', 'cloud', metrics_list)
        self._CheckParameterValue('Num Processes', '1', metrics_list)
        self._CheckParameterValue('Num Threads', '1', metrics_list)
        self._CheckParameterValue('Provider Types', bucket1_uri.scheme, metrics_list)
        self._CheckParameterValue('Number of Files/Objects Transferred', '1', metrics_list)
        self._CheckParameterValue('Size of Files/Objects Transferred', file_size, metrics_list)

    @unittest.skipUnless(HAS_S3_CREDS, 'Test requires both S3 and GS credentials')
    def testCrossProviderDaisyChainCollection(self):
        """Tests the collection of daisy-chain operations."""
        s3_bucket = self.CreateBucket(provider='s3')
        gs_bucket = self.CreateBucket(provider='gs')
        unused_s3_key = self.CreateObject(bucket_uri=s3_bucket, contents=b'foo')
        gs_key = self.CreateObject(bucket_uri=gs_bucket, contents=b'bar')
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['rsync', suri(s3_bucket), suri(gs_bucket)])
        self._CheckParameterValue('Event Action', 'CloudToCloud%2CDaisyChain', metrics_list)
        self._CheckParameterValue('Provider Types', 'gs%2Cs3', metrics_list)
        metrics_list = self._RunGsUtilWithAnalyticsOutput(['cp', suri(gs_key), suri(s3_bucket)])
        self._CheckParameterValue('Event Action', 'CloudToCloud%2CDaisyChain', metrics_list)
        self._CheckParameterValue('Provider Types', 'gs%2Cs3', metrics_list)