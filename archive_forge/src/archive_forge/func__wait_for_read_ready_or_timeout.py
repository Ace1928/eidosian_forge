import locale
import logging
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from .termhelpers import Nonblocking
from . import events
from typing import (
from types import TracebackType, FrameType
def _wait_for_read_ready_or_timeout(self, timeout: Union[float, int, None]) -> Tuple[bool, Optional[Union[events.Event, str]]]:
    """Returns tuple of whether stdin is ready to read and an event.

        If an event is returned, that event is more pressing than reading
        bytes on stdin to create a keyboard input event.
        If stdin is ready, either there are bytes to read or a SIGTSTP
        triggered by dsusp has been received"""
    remaining_timeout = timeout
    t0 = time.time()
    while True:
        try:
            rs, _, _ = select.select([self.in_stream.fileno()] + ([] if self.wakeup_read_fd is None else [self.wakeup_read_fd]) + self.readers, [], [], remaining_timeout)
            if not rs:
                return (False, None)
            r = rs[0]
            if r == self.in_stream.fileno():
                return (True, None)
            elif r == self.wakeup_read_fd:
                signal_number = ord(os.read(r, 1))
                if signal_number == signal.SIGINT:
                    raise InterruptedError()
            else:
                os.read(r, 1024)
                if self.queued_interrupting_events:
                    return (False, self.queued_interrupting_events.pop(0))
                elif remaining_timeout is not None:
                    remaining_timeout = max(0, t0 + remaining_timeout - time.time())
                    continue
                else:
                    continue
        except OSError:
            if self.sigints:
                return (False, self.sigints.pop())
            if remaining_timeout is not None:
                remaining_timeout = max(remaining_timeout - (time.time() - t0), 0)