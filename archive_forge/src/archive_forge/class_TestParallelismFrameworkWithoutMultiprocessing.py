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
class TestParallelismFrameworkWithoutMultiprocessing(TestParallelismFramework):
    """Tests parallelism framework works with multiprocessing module unavailable.

  Notably, this test has no way to override previous calls
  to gslib.util.CheckMultiprocessingAvailableAndInit to prevent the
  initialization of all of the global variables in command.py, so this still
  behaves slightly differently than the behavior one would see on a machine
  where the multiprocessing functionality is actually not available (in
  particular, it will not catch the case where a global variable that is not
  available for the sequential path is referenced before initialization).
  """
    command_class = FakeCommandWithoutMultiprocessingModule