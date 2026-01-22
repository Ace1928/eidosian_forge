from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _GetAnsiCursorUpSequence(self, num_lines):
    """Returns an ANSI control sequences that moves the cursor up num_lines."""
    return '\x1b[{}A'.format(num_lines)