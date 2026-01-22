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
class TestReporter(TestCase):
    """
    Tests for L{Reporter}.
    """

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

    def test_syntaxErrorNoOffset(self):
        """
        C{syntaxError} doesn't include a caret pointing to the error if
        C{offset} is passed as C{None}.
        """
        err = io.StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, None, 'bad line of source')
        self.assertEqual('foo.py:3: a problem\nbad line of source\n', err.getvalue())

    def test_syntaxErrorNoText(self):
        """
        C{syntaxError} doesn't include text or nonsensical offsets if C{text} is C{None}.

        This typically happens when reporting syntax errors from stdin.
        """
        err = io.StringIO()
        reporter = Reporter(None, err)
        reporter.syntaxError('<stdin>', 'a problem', 0, 0, None)
        self.assertEqual('<stdin>:1:1: a problem\n', err.getvalue())

    def test_multiLineSyntaxError(self):
        """
        If there's a multi-line syntax error, then we only report the last
        line.  The offset is adjusted so that it is relative to the start of
        the last line.
        """
        err = io.StringIO()
        lines = ['bad line of source', 'more bad lines of source']
        reporter = Reporter(None, err)
        reporter.syntaxError('foo.py', 'a problem', 3, len(lines[0]) + 7, '\n'.join(lines))
        self.assertEqual('foo.py:3:25: a problem\n' + lines[-1] + '\n' + ' ' * 24 + '^\n', err.getvalue())

    def test_unexpectedError(self):
        """
        C{unexpectedError} reports an error processing a source file.
        """
        err = io.StringIO()
        reporter = Reporter(None, err)
        reporter.unexpectedError('source.py', 'error message')
        self.assertEqual('source.py: error message\n', err.getvalue())

    def test_flake(self):
        """
        C{flake} reports a code warning from Pyflakes.  It is exactly the
        str() of a L{pyflakes.messages.Message}.
        """
        out = io.StringIO()
        reporter = Reporter(out, None)
        message = UnusedImport('foo.py', Node(42), 'bar')
        reporter.flake(message)
        self.assertEqual(out.getvalue(), f'{message}\n')