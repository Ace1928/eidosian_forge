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
class ReplacedSigIntHandler(ContextManager):

    def __init__(self, handler: Callable) -> None:
        self.handler = handler

    def __enter__(self) -> None:
        self.orig_sigint_handler = signal.signal(signal.SIGINT, self.handler)

    def __exit__(self, type: Optional[Type[BaseException]]=None, value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        signal.signal(signal.SIGINT, self.orig_sigint_handler)