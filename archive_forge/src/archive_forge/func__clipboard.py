from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable
from prompt_toolkit.selection import SelectionType
def _clipboard(self) -> Clipboard:
    return self.get_clipboard() or DummyClipboard()