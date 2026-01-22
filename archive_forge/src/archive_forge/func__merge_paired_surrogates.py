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
@staticmethod
def _merge_paired_surrogates(key_presses: list[KeyPress]) -> Iterator[KeyPress]:
    """
        Combines consecutive KeyPresses with high and low surrogates into
        single characters
        """
    buffered_high_surrogate = None
    for key in key_presses:
        is_text = not isinstance(key.key, Keys)
        is_high_surrogate = is_text and '\ud800' <= key.key <= '\udbff'
        is_low_surrogate = is_text and '\udc00' <= key.key <= '\udfff'
        if buffered_high_surrogate:
            if is_low_surrogate:
                fullchar = (buffered_high_surrogate.key + key.key).encode('utf-16-le', 'surrogatepass').decode('utf-16-le')
                key = KeyPress(fullchar, fullchar)
            else:
                yield buffered_high_surrogate
            buffered_high_surrogate = None
        if is_high_surrogate:
            buffered_high_surrogate = key
        else:
            yield key
    if buffered_high_surrogate:
        yield buffered_high_surrogate