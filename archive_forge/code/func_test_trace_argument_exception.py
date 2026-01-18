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
def test_trace_argument_exception(self):
    err = Exception('an error')
    mutter('can format stringable classes %s', err)
    log = self.get_log()
    self.assertContainsRe(log, 'can format stringable classes an error')