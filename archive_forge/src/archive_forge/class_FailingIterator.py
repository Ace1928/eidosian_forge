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
class FailingIterator(six.Iterator):

    def __init__(self, size, failure_indices):
        self.size = size
        self.failure_indices = failure_indices
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index == self.size:
            raise StopIteration('')
        elif self.current_index in self.failure_indices:
            self.current_index += 1
            raise CustomException('Iterator failing on purpose at index %d.' % self.current_index)
        else:
            self.current_index += 1
            return self.current_index - 1