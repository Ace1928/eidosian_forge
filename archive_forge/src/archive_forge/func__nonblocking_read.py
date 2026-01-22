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
def _nonblocking_read(self) -> int:
    """Returns the number of characters read and adds them to self.unprocessed_bytes"""
    with Nonblocking(self.in_stream):
        try:
            data = os.read(self.in_stream.fileno(), READ_SIZE)
        except BlockingIOError:
            return 0
        if data:
            self.unprocessed_bytes.extend((data[i:i + 1] for i in range(len(data))))
            return len(data)
        else:
            return 0