from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import os
import signal
import sys
import threading
import time
import enum
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import multiline
from googlecloudsdk.core.console.style import parser
import six
class _NonInteractiveProgressTracker(_BaseProgressTracker):
    """A context manager for telling the user about long-running progress."""

    def Tick(self):
        """Give a visual indication to the user that some progress has been made.

    Output is sent to sys.stderr. Nothing is shown if output is not a TTY.

    Returns:
      Whether progress has completed.
    """
        with self._lock:
            if not self._done:
                self._Print('.')
        return self._done

    def _Print(self, message=''):
        """Reprints the prefix followed by an optional message.

    If there is a multiline message, we print the full message and every
    time the Prefix Message is the same, we only reprint the last line to
    account for a different 'message'. If there is a new message, we print
    on a new line.

    Args:
      message: str, suffix of message
    """
        if self._spinner_only or not self._output_enabled:
            return
        display_message = self._GetPrefix()
        self._stream.write(message or display_message + '\n')
        return