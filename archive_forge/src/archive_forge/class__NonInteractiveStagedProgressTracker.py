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
class _NonInteractiveStagedProgressTracker(_NormalStagedProgressTracker):
    """A context manager for telling the user about long-running progress."""

    def _SetupExitOutput(self):
        """Sets up output to print out the closing line."""
        return

    def _SetupOutput(self):
        self._Print(self._message + '\n')

    def _GetTickMark(self, ticks):
        """Returns the next tick mark."""
        return '.'

    def _GetStagedCompletedSuffix(self, status):
        return status.value + '\n'

    def _SetUpOutputForStage(self, stage):
        message = stage.header
        if stage.message:
            message += ' ' + stage.message + '...'
        self._Print(message)

    def _Print(self, message=''):
        """Prints an update containing message to the output stream.

    Args:
      message: str, suffix of message
    """
        if not self._output_enabled:
            return
        self._stream.write(message)