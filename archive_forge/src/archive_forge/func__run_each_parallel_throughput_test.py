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
def _run_each_parallel_throughput_test(self, test_name, num_processes, num_threads, compression_ratio=None):
    self._run_throughput_test(test_name, num_processes, num_threads, 'fan', compression_ratio=compression_ratio)
    if not RUN_S3_TESTS:
        self._run_throughput_test(test_name, num_processes, num_threads, 'slice', compression_ratio=compression_ratio)
        self._run_throughput_test(test_name, num_processes, num_threads, 'both', compression_ratio=compression_ratio)