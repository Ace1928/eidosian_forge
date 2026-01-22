from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from functools import wraps
import os.path
import random
import re
import shutil
import tempfile
import six
import boto
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.utils.constants import UTF8
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
def assertRegexpMatchesWithFlags(self, text, pattern, msg=None, flags=0):
    """Like assertRegexpMatches, but allows specifying additional re flags.

    Args:
      text: The text in which to search for pattern.
      pattern: The pattern to search for; should be either a string or compiled
          regex returned from re.compile().
      msg: The message to be displayed if pattern is not found in text. The
          values for pattern and text will be included after this message.
      flags: Additional flags from the re module to be included when compiling
          pattern. If pattern is a regex that was compiled with existing flags,
          these, flags will be added via a bitwise-or.
    """
    if isinstance(pattern, six.string_types):
        pattern = re.compile(pattern, flags=flags)
    else:
        pattern = re.compile(pattern.pattern, flags=pattern.flags | flags)
    if not pattern.search(text):
        failure_msg = msg or "Regex didn't match"
        failure_msg = '%s: %r not found in %r' % (failure_msg, pattern.pattern, text)
        raise self.failureException(failure_msg)