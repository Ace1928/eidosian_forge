from __future__ import unicode_literals
from ..terminal.win32_input import ConsoleInputReader
from ..win32_types import SECURITY_ATTRIBUTES
from .base import EventLoop, INPUT_TIMEOUT
from .inputhook import InputHookContext
from .utils import TimeIt
from ctypes import windll, pointer
from ctypes.wintypes import DWORD, BOOL, HANDLE
import msvcrt
import threading
def _process_queued_calls_from_executor(self):
    calls_from_executor, self._calls_from_executor = (self._calls_from_executor, [])
    for c in calls_from_executor:
        c()