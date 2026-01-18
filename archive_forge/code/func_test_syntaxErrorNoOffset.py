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
def test_syntaxErrorNoOffset(self):
    """
        C{syntaxError} doesn't include a caret pointing to the error if
        C{offset} is passed as C{None}.
        """
    err = io.StringIO()
    reporter = Reporter(None, err)
    reporter.syntaxError('foo.py', 'a problem', 3, None, 'bad line of source')
    self.assertEqual('foo.py:3: a problem\nbad line of source\n', err.getvalue())