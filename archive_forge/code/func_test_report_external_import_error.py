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
def test_report_external_import_error(self):
    """Short friendly message for missing system modules."""
    try:
        import ImaginaryModule
    except ImportError:
        msg = _format_exception()
    else:
        self.fail('somehow succeeded in importing %r' % ImaginaryModule)
    self.assertContainsRe(msg, "^brz: ERROR: No module named '?ImaginaryModule'?\nYou may need to install this Python library separately.\n$")