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
def test_push_log_file(self):
    """Can push and pop log file, and this catches mutter messages.

        This is primarily for use in the test framework.
        """
    tmp1 = tempfile.NamedTemporaryFile()
    tmp2 = tempfile.NamedTemporaryFile()
    try:
        memento1 = push_log_file(tmp1)
        mutter('comment to file1')
        try:
            memento2 = push_log_file(tmp2)
            try:
                mutter('comment to file2')
            finally:
                pop_log_file(memento2)
            mutter('again to file1')
        finally:
            pop_log_file(memento1)
        tmp1.seek(0)
        self.assertContainsRe(tmp1.read(), b'\\d+\\.\\d+  comment to file1\n\\d+\\.\\d+  again to file1\n')
        tmp2.seek(0)
        self.assertContainsRe(tmp2.read(), b'\\d+\\.\\d+  comment to file2\n')
    finally:
        tmp1.close()
        tmp2.close()