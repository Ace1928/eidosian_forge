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
def PrintSkippedTests(self, sequential_skipped=set(), parallel_skipped=set()):
    """Prints all skipped tests, and the reasons they  were skipped.

    Takes the union of sequentual_skipped and parallel_skipped,
    and pretty-prints the resulting methods and reasons. Note that these two
    arguments are lists of tuples from TestResult.skipped as described here:
    https://docs.python.org/2/library/unittest.html#unittest.TestResult.skipped

    Args:
        sequentual_skipped: An instance of TestResult.skipped.
        parallel_skipped: An instance of TestResult.skipped.
    """
    if len(sequential_skipped) > 0 or len(parallel_skipped) > 0:
        sequential_skipped = set(sequential_skipped)
        parallel_skipped = set(parallel_skipped)
        all_skipped = sequential_skipped.union(parallel_skipped)
        print('Tests skipped:')
        for method, reason in all_skipped:
            print('  ' + method.id())
            print('    Reason: ' + reason)