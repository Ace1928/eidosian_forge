from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core.util import encoding
Given a stacktrace, only include Cloud SDK files in path.

  Args:
    traceback: str, the original unformatted traceback

  Returns:
    str, A new stacktrace with the private paths removed
    None, If traceback does not match traceback pattern
  