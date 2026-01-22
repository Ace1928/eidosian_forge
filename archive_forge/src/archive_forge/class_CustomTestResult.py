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
class CustomTestResult(unittest.TextTestResult):
    """A subclass of unittest.TextTestResult that prints a progress report."""

    def startTest(self, test):
        super(CustomTestResult, self).startTest(test)
        if self.dots:
            test_id = '.'.join(test.id().split('.')[-2:])
            message = '\r%d/%d finished - E[%d] F[%d] s[%d] - %s' % (self.testsRun, total_tests, len(self.errors), len(self.failures), len(self.skipped), test_id)
            message = message[:73]
            message = message.ljust(73)
            self.stream.write('%s - ' % message)