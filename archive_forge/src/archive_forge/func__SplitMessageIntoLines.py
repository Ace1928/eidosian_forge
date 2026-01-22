from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _SplitMessageIntoLines(self, message):
    """Converts message into a list of strs, each representing a line."""
    lines = self._console_attr.SplitLine(message, self.effective_width)
    for i in range(len(lines)):
        lines[i] += '\n'
    return lines