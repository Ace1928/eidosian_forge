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
def test_trace_argument_unicode(self):
    """Write a Unicode argument to the trace log"""
    mutter('the unicode character for benzene is %s', '⌬')
    log = self.get_log()
    self.assertContainsRe(log, 'the unicode character')