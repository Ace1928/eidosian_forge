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
def test_report_import_syntax_error(self):
    try:
        raise ImportError('syntax error')
    except ImportError:
        msg = _format_exception()
    self.assertContainsRe(msg, 'Breezy has encountered an internal error')