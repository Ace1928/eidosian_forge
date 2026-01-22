from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import logging
import os
import subprocess
import re
import sys
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import range
import gslib
from gslib.cloud_api import ProjectIdException
from gslib.command import Command
from gslib.command import ResetFailureCount
from gslib.exception import CommandException
from gslib.project_id import PopulateProjectId
import gslib.tests as tests
from gslib.tests.util import GetTestNames
from gslib.tests.util import InvokedFromParFile
from gslib.tests.util import unittest
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.system_util import IS_WINDOWS
def PrintTestResults(self, num_sequential_tests, sequential_success, sequential_skipped, sequential_time_elapsed, num_parallel_tests, num_parallel_failures, parallel_time_elapsed):
    """Prints test results for parallel and sequential tests."""
    print('Parallel tests complete. Success: %s Fail: %s' % (num_parallel_tests - num_parallel_failures, num_parallel_failures))
    print('Ran %d tests in %.3fs (%d sequential in %.3fs, %d parallel in %.3fs)' % (num_parallel_tests + num_sequential_tests, float(sequential_time_elapsed + parallel_time_elapsed), num_sequential_tests, float(sequential_time_elapsed), num_parallel_tests, float(parallel_time_elapsed)))
    self.PrintSkippedTests(sequential_skipped)
    print()
    if not num_parallel_failures and sequential_success:
        print('OK')
    else:
        if num_parallel_failures:
            print('FAILED (parallel tests)')
        if not sequential_success:
            print('FAILED (sequential tests)')