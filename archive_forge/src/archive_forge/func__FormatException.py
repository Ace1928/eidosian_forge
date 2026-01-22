from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import traceback
from googlecloudsdk.core.util import encoding
import six
def _FormatException(exc_type, exc, exc_trace):
    """Returns a formatted exception message from an exception and traceback."""
    exc_msg_lines = []
    for line in traceback.format_exception(exc_type, exc, exc_trace):
        exc_msg_lines.append(encoding.Decode(line))
    return ''.join(exc_msg_lines)