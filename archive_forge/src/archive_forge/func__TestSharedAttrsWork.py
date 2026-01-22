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