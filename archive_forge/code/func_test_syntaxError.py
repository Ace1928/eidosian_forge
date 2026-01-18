import contextlib
import io
import os
import sys
import shutil
import subprocess
import tempfile
from pyflakes.checker import PYPY
from pyflakes.messages import UnusedImport
from pyflakes.reporter import Reporter
from pyflakes.api import (
from pyflakes.test.harness import TestCase, skipIf
def test_syntaxError(self):
    """
        C{syntaxError} reports that there was a syntax error in the source
        file.  It reports to the error stream and includes the filename, line
        number, error message, actual line of source and a caret pointing to
        where the error is.
        """
    err = io.StringIO()
    reporter = Reporter(None, err)
    reporter.syntaxError('foo.py', 'a problem', 3, 8, 'bad line of source')
    self.assertEqual('foo.py:3:8: a problem\nbad line of source\n       ^\n', err.getvalue())