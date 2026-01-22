from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import socket
import sys
import six
import boto
from gslib.commands.perfdiag import _GenerateFileData
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForXML
from gslib.tests.util import ObjectToURI as suri
from gslib.tests.util import RUN_S3_TESTS
from gslib.tests.util import unittest
from gslib.utils.system_util import IS_WINDOWS
from six import add_move, MovedModule
from six.moves import mock
class TestPerfDiag(testcase.GsUtilIntegrationTestCase):
    """Integration tests for perfdiag command."""

    @classmethod
    def setUpClass(cls):
        super(TestPerfDiag, cls).setUpClass()
        gs_host = boto.config.get('Credentials', 'gs_host', boto.gs.connection.GSConnection.DefaultHost)
        gs_ip = None
        for address_tuple in socket.getaddrinfo(gs_host, None):
            if address_tuple[0].name in ('AF_INET', 'AF_INET6'):
                gs_ip = address_tuple[4][0]
                break
        if not gs_ip:
            raise ConnectionError('Count not find IP for ' + gs_host)
        cls._custom_endpoint_flags = ['-o', 'Credentials:gs_host=' + gs_ip, '-o', 'Credentials:gs_host_header=' + gs_host, '-o', 'Boto:https_validate_certificates=False']

    def _should_run_with_custom_endpoints(self):
        python_version_less_than_2_7_9 = sys.version_info[0] == 2 and (sys.version_info[1] < 7 or (sys.version_info[1] == 7 and sys.version_info[2] < 9))
        return self.test_api == 'XML' and (not RUN_S3_TESTS) and python_version_less_than_2_7_9 and (not (os.environ.get('http_proxy') or os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')))

    def test_latency(self):
        bucket_uri = self.CreateBucket()
        cmd = ['perfdiag', '-n', '1', '-t', 'lat', suri(bucket_uri)]
        self.RunGsUtil(cmd)
        if self._should_run_with_custom_endpoints():
            self.RunGsUtil(self._custom_endpoint_flags + cmd)
        self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)

    def _run_throughput_test(self, test_name, num_processes, num_threads, parallelism_strategy=None, compression_ratio=None):
        bucket_uri = self.CreateBucket()
        cmd = ['perfdiag', '-n', str(num_processes * num_threads), '-s', '1024', '-c', str(num_processes), '-k', str(num_threads), '-t', test_name]
        if compression_ratio is not None:
            cmd += ['-j', str(compression_ratio)]
        if parallelism_strategy is not None:
            cmd += ['-p', parallelism_strategy]
        cmd += [suri(bucket_uri)]
        stderr_default = self.RunGsUtil(cmd, return_stderr=True)
        stderr_custom = None
        if self._should_run_with_custom_endpoints():
            stderr_custom = self.RunGsUtil(self._custom_endpoint_flags + cmd, return_stderr=True)
        self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)
        return (stderr_default, stderr_custom)

    def _run_each_parallel_throughput_test(self, test_name, num_processes, num_threads, compression_ratio=None):
        self._run_throughput_test(test_name, num_processes, num_threads, 'fan', compression_ratio=compression_ratio)
        if not RUN_S3_TESTS:
            self._run_throughput_test(test_name, num_processes, num_threads, 'slice', compression_ratio=compression_ratio)
            self._run_throughput_test(test_name, num_processes, num_threads, 'both', compression_ratio=compression_ratio)

    def test_write_throughput_single_process_single_thread(self):
        self._run_throughput_test('wthru', 1, 1)
        self._run_throughput_test('wthru_file', 1, 1)

    def test_write_throughput_single_process_multi_thread(self):
        self._run_each_parallel_throughput_test('wthru', 1, 2)
        self._run_each_parallel_throughput_test('wthru_file', 1, 2)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def test_write_throughput_multi_process_single_thread(self):
        self._run_each_parallel_throughput_test('wthru', 2, 1)
        self._run_each_parallel_throughput_test('wthru_file', 2, 1)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def test_write_throughput_multi_process_multi_thread(self):
        self._run_each_parallel_throughput_test('wthru', 2, 2)
        self._run_each_parallel_throughput_test('wthru_file', 2, 2)

    def test_read_throughput_single_process_single_thread(self):
        self._run_throughput_test('rthru', 1, 1)
        self._run_throughput_test('rthru_file', 1, 1)

    def test_read_throughput_single_process_multi_thread(self):
        self._run_each_parallel_throughput_test('rthru', 1, 2)
        self._run_each_parallel_throughput_test('rthru_file', 1, 2)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def test_read_throughput_multi_process_single_thread(self):
        self._run_each_parallel_throughput_test('rthru', 2, 1)
        self._run_each_parallel_throughput_test('rthru_file', 2, 1)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def test_read_throughput_multi_process_multi_thread(self):
        self._run_each_parallel_throughput_test('rthru', 2, 2)
        self._run_each_parallel_throughput_test('rthru_file', 2, 2)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def test_read_and_write_file_ordering(self):
        """Tests that rthru_file and wthru_file work when run together."""
        self._run_throughput_test('rthru_file,wthru_file', 1, 1)
        self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'fan')
        if not RUN_S3_TESTS:
            self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'slice')
            self._run_throughput_test('rthru_file,wthru_file', 2, 2, 'both')

    def test_input_output(self):
        outpath = self.CreateTempFile()
        bucket_uri = self.CreateBucket()
        self.RunGsUtil(['perfdiag', '-o', outpath, '-n', '1', '-t', 'lat', suri(bucket_uri)])
        self.RunGsUtil(['perfdiag', '-i', outpath])

    def test_invalid_size(self):
        stderr = self.RunGsUtil(['perfdiag', '-n', '1', '-s', 'foo', '-t', 'wthru', 'gs://foobar'], expected_status=1, return_stderr=True)
        self.assertIn('Invalid -s', stderr)

    def test_toobig_size(self):
        stderr = self.RunGsUtil(['perfdiag', '-n', '1', '-s', '3pb', '-t', 'wthru', 'gs://foobar'], expected_status=1, return_stderr=True)
        self.assertIn('in-memory tests maximum file size', stderr)

    def test_listing(self):
        bucket_uri = self.CreateBucket()
        stdout = self.RunGsUtil(['perfdiag', '-n', '1', '-t', 'list', suri(bucket_uri)], return_stdout=True)
        self.assertIn('Number of listing calls made:', stdout)
        self.AssertNObjectsInBucket(bucket_uri, 0, versioned=True)

    @SkipForXML('No compressed transport encoding support for the XML API.')
    def test_gzip_write_throughput_single_process_single_thread(self):
        stderr_default, _ = self._run_throughput_test('wthru', 1, 1, compression_ratio=50)
        self.assertIn('Gzip compression ratio: 50', stderr_default)
        self.assertIn('Gzip transport encoding writes: True', stderr_default)
        stderr_default, _ = self._run_throughput_test('wthru_file', 1, 1, compression_ratio=50)
        self.assertIn('Gzip compression ratio: 50', stderr_default)
        self.assertIn('Gzip transport encoding writes: True', stderr_default)

    @SkipForXML('No compressed transport encoding support for the XML API.')
    def test_gzip_write_throughput_single_process_multi_thread(self):
        self._run_each_parallel_throughput_test('wthru', 1, 2, compression_ratio=50)
        self._run_each_parallel_throughput_test('wthru_file', 1, 2, compression_ratio=50)

    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    @SkipForXML('No compressed transport encoding support for the XML API.')
    def test_gzip_write_throughput_multi_process_multi_thread(self):
        self._run_each_parallel_throughput_test('wthru', 2, 2, compression_ratio=50)
        self._run_each_parallel_throughput_test('wthru_file', 2, 2, compression_ratio=50)