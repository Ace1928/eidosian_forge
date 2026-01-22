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
class _BaseProgressTracker(six.with_metaclass(abc.ABCMeta, object)):
    """A context manager for telling the user about long-running progress."""

    def __init__(self, message, autotick, detail_message_callback, done_message_callback, tick_delay, interruptable, aborted_message, spinner_override_message, no_spacing):
        self._stream = sys.stderr
        if message is None:
            self._spinner_only = True
            self._message = ''
            self._prefix = ''
        else:
            self._spinner_only = False
            self._message = message
            self._prefix = message + ('' if no_spacing else '...')
        self._detail_message_callback = detail_message_callback
        self.spinner_override_message = spinner_override_message
        self._done_message_callback = done_message_callback
        self._ticks = 0
        self._done = False
        self._lock = threading.Lock()
        self._tick_delay = tick_delay
        self._ticker = None
        console_width = console_attr.ConsoleAttr().GetTermSize()[0]
        if console_width < 0:
            console_width = 0
        self._output_enabled = log.IsUserOutputEnabled() and console_width != 0
        self.__autotick = autotick and self._output_enabled
        self._interruptable = interruptable
        self._aborted_message = aborted_message
        self._old_signal_handler = None
        self._symbols = console_attr.GetConsoleAttr().GetProgressTrackerSymbols()
        self._no_spacing = no_spacing
        self._is_tty = console_io.IsInteractive(error=True)

    @property
    def _autotick(self):
        return self.__autotick

    def _GetPrefix(self):
        if self._is_tty and self._detail_message_callback:
            detail_message = self._detail_message_callback()
            if detail_message:
                if self._no_spacing:
                    return self._prefix + detail_message
                return self._prefix + ' ' + detail_message + '...'
        return self._prefix

    def _SetUpSignalHandler(self):
        """Sets up a signal handler for handling SIGINT."""

        def _CtrlCHandler(unused_signal, unused_frame):
            if self._interruptable:
                raise console_io.OperationCancelledError(self._aborted_message)
            else:
                with self._lock:
                    sys.stderr.write('\n\nThis operation cannot be cancelled.\n\n')
        try:
            self._old_signal_handler = signal.signal(signal.SIGINT, _CtrlCHandler)
            self._restore_old_handler = True
        except ValueError:
            self._restore_old_handler = False

    def _TearDownSignalHandler(self):
        if self._restore_old_handler:
            try:
                signal.signal(signal.SIGINT, self._old_signal_handler)
            except ValueError:
                pass

    def __enter__(self):
        self._SetUpSignalHandler()
        log.file_only_logger.info(self._GetPrefix())
        self._Print()
        if self._autotick:

            def Ticker():
                while True:
                    _SleepSecs(self._tick_delay)
                    if self.Tick():
                        return
            self._ticker = threading.Thread(target=Ticker)
            self._ticker.start()
        return self

    def __exit__(self, unused_ex_type, exc_value, unused_traceback):
        with self._lock:
            self._done = True
            if exc_value:
                if isinstance(exc_value, console_io.OperationCancelledError):
                    self._Print('aborted by ctrl-c.\n')
                else:
                    self._Print('failed.\n')
            elif not self._spinner_only:
                if self._done_message_callback:
                    self._Print(self._done_message_callback())
                else:
                    self._Print('done.\n')
        if self._ticker:
            self._ticker.join()
        self._TearDownSignalHandler()

    @abc.abstractmethod
    def Tick(self):
        """Give a visual indication to the user that some progress has been made.

    Output is sent to sys.stderr. Nothing is shown if output is not a TTY.

    Returns:
      Whether progress has completed.
    """
        pass

    @abc.abstractmethod
    def _Print(self, message=''):
        """Prints an update containing message to the output stream."""
        pass