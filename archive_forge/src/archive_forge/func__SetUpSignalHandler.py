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
def _SetUpSignalHandler(self):
    """Sets up a signal handler for handling SIGINT."""

    def _CtrlCHandler(unused_signal, unused_frame):
        if self._interruptable:
            raise console_io.OperationCancelledError(self._aborted_message)
        else:
            self._NotifyUninterruptableError()
    try:
        self._old_signal_handler = signal.signal(signal.SIGINT, _CtrlCHandler)
        self._restore_old_handler = True
    except ValueError:
        self._restore_old_handler = False