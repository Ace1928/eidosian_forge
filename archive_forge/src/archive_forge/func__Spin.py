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
def _Spin(self, unused_sig=None, unused_frame=None):
    """Rotates the spinner one tick and checks for timeout."""
    self._ticks += 1
    self._WriteMark(self._symbols.spin_marks[self._ticks % len(self._symbols.spin_marks)])
    self._timeout -= self._tick_delay
    if not self._TimedOut():
        return
    self._WriteMark('?')
    if os.fork():
        os._exit(1)
    signal.signal(signal.SIGALRM, self._ExitBackground)
    signal.setitimer(signal.ITIMER_REAL, self._background_ttl, self._background_ttl)
    self._ofile = None