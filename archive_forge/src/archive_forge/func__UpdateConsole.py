from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import threading
from googlecloudsdk.core.console import console_attr
import six
def _UpdateConsole(self):
    """Updates the console output to show any updated or added messages."""
    if not self._may_have_update:
        return
    if self._last_total_lines:
        self._stream.write(self._GetAnsiCursorUpSequence(self._last_total_lines))
    total_lines = 0
    force_print_rest = False
    for message in self._messages:
        num_lines = message.num_lines
        total_lines += num_lines
        if message.has_update or force_print_rest:
            force_print_rest |= message.num_lines_changed
            message.Print()
        else:
            self._stream.write('\n' * num_lines)
    self._last_total_lines = total_lines
    self._may_have_update = False