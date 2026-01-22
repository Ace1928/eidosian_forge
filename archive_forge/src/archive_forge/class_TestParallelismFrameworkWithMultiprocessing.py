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
class TestParallelismFrameworkWithMultiprocessing(testcase.GsUtilUnitTestCase):
    """Tests that only run with multiprocessing enabled."""

    @RequiresIsolation
    @mock.patch.object(FakeCommand, '_ResetConnectionPool', side_effect=functools.partial(call_queue.put, None))
    @unittest.skipIf(IS_WINDOWS, 'Multiprocessing is not supported on Windows')
    def testResetConnectionPoolCalledOncePerProcess(self, mock_reset_connection_pool):
        expected_call_count = 2
        FakeCommand(True).Apply(_ReturnOneValue, [1, 2, 3], _ExceptionHandler, process_count=expected_call_count, thread_count=3, arg_checker=DummyArgChecker)
        for _ in range(expected_call_count):
            self.assertIsNone(call_queue.get(timeout=1.0))