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
def assertRaisesFireExit(self, code, regexp='.*'):
    """Asserts that a FireExit error is raised in the context.

    Allows tests to check that Fire's wrapper around SystemExit is raised
    and that a regexp is matched in the output.

    Args:
      code: The status code that the FireExit should contain.
      regexp: stdout must match this regex.

    Yields:
      Yields to the wrapped context.
    """
    with self.assertOutputMatches(stderr=regexp):
        with self.assertRaises(core.FireExit):
            try:
                yield
            except core.FireExit as exc:
                if exc.code != code:
                    raise AssertionError('Incorrect exit code: %r != %r' % (exc.code, code))
                self.assertIsInstance(exc.trace, trace.FireTrace)
                raise