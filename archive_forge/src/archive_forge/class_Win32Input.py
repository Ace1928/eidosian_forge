from __future__ import annotations
import os
import sys
from abc import abstractmethod
from asyncio import get_running_loop
from contextlib import contextmanager
from ..utils import SPHINX_AUTODOC_RUNNING
from ctypes import Array, pointer
from ctypes.wintypes import DWORD, HANDLE
from typing import Callable, ContextManager, Iterable, Iterator, TextIO
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.eventloop.win32 import create_win32_event, wait_for_handles
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.win32_types import (
from .ansi_escape_sequences import REVERSE_ANSI_SEQUENCES
from .base import Input
class Win32Input(_Win32InputBase):
    """
    `Input` class that reads from the Windows console.
    """

    def __init__(self, stdin: TextIO | None=None) -> None:
        super().__init__()
        self.console_input_reader = ConsoleInputReader()

    def attach(self, input_ready_callback: Callable[[], None]) -> ContextManager[None]:
        """
        Return a context manager that makes this input active in the current
        event loop.
        """
        return attach_win32_input(self, input_ready_callback)

    def detach(self) -> ContextManager[None]:
        """
        Return a context manager that makes sure that this input is not active
        in the current event loop.
        """
        return detach_win32_input(self)

    def read_keys(self) -> list[KeyPress]:
        return list(self.console_input_reader.read())

    def flush(self) -> None:
        pass

    @property
    def closed(self) -> bool:
        return False

    def raw_mode(self) -> ContextManager[None]:
        return raw_mode()

    def cooked_mode(self) -> ContextManager[None]:
        return cooked_mode()

    def fileno(self) -> int:
        return sys.stdin.fileno()

    def typeahead_hash(self) -> str:
        return 'win32-input'

    def close(self) -> None:
        self.console_input_reader.close()

    @property
    def handle(self) -> HANDLE:
        return self.console_input_reader.handle