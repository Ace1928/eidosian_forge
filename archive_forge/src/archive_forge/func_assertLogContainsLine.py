import errno
import logging
import os
import re
import sys
import tempfile
from io import StringIO
from .. import debug, errors, trace
from ..trace import (_rollover_trace_maybe, be_quiet, get_verbosity_level,
from . import TestCase, TestCaseInTempDir, TestSkipped, features
def assertLogContainsLine(self, log, string):
    """Assert log contains a line including log timestamp."""
    self.assertContainsRe(log, '(?m)^\\d+\\.\\d+  ' + re.escape(string))