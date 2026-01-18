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
def test_report_broken_pipe(self):
    try:
        raise OSError(errno.EPIPE, 'broken pipe foofofo')
    except OSError:
        msg = _format_exception()
        self.assertEqual(msg, 'brz: broken pipe\n')
    else:
        self.fail('expected error not raised')