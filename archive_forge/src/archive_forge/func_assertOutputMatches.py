from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import os
import re
import sys
import unittest
from fire import core
from fire import trace
import mock
import six
@contextlib.contextmanager
def assertOutputMatches(self, stdout='.*', stderr='.*', capture=True):
    """Asserts that the context generates stdout and stderr matching regexps.

    Note: If wrapped code raises an exception, stdout and stderr will not be
      checked.

    Args:
      stdout: (str) regexp to match against stdout (None will check no stdout)
      stderr: (str) regexp to match against stderr (None will check no stderr)
      capture: (bool, default True) do not bubble up stdout or stderr

    Yields:
      Yields to the wrapped context.
    """
    stdout_fp = six.StringIO()
    stderr_fp = six.StringIO()
    try:
        with mock.patch.object(sys, 'stdout', stdout_fp):
            with mock.patch.object(sys, 'stderr', stderr_fp):
                yield
    finally:
        if not capture:
            sys.stdout.write(stdout_fp.getvalue())
            sys.stderr.write(stderr_fp.getvalue())
    for name, regexp, fp in [('stdout', stdout, stdout_fp), ('stderr', stderr, stderr_fp)]:
        value = fp.getvalue()
        if regexp is None:
            if value:
                raise AssertionError('%s: Expected no output. Got: %r' % (name, value))
        elif not re.search(regexp, value, re.DOTALL | re.MULTILINE):
            raise AssertionError('%s: Expected %r to match %r' % (name, value, regexp))