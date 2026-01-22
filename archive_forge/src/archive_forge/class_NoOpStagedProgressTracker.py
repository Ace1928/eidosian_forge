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
class NoOpStagedProgressTracker(_BaseStagedProgressTracker):
    """A staged progress tracker that doesn't do anything."""

    def __init__(self, stages, interruptable=False, aborted_message=''):
        super(NoOpStagedProgressTracker, self).__init__(message='', stages=stages, success_message='', warning_message='', failure_message='', autotick=False, tick_delay=0, interruptable=interruptable, aborted_message=aborted_message, tracker_id='', done_message_callback=None, console=console_attr.ConsoleAttr(encoding='ascii', suppress_output=True))
        self._aborted_message = aborted_message
        self._done = False

    def __enter__(self):

        def _CtrlCHandler(unused_signal, unused_frame):
            if self._interruptable:
                raise console_io.OperationCancelledError(self._aborted_message)
        self._old_signal_handler = signal.signal(signal.SIGINT, _CtrlCHandler)
        return self

    def _Print(self, message=''):
        return

    def Tick(self):
        return self._done

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._done = True
        signal.signal(signal.SIGINT, self._old_signal_handler)

    def _SetupOutput(self):
        pass

    def UpdateHeaderMessage(self, message):
        pass