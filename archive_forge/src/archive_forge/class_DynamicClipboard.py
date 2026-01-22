from __future__ import annotations
from abc import ABCMeta, abstractmethod
from typing import Callable
from prompt_toolkit.selection import SelectionType
class DynamicClipboard(Clipboard):
    """
    Clipboard class that can dynamically returns any Clipboard.

    :param get_clipboard: Callable that returns a :class:`.Clipboard` instance.
    """

    def __init__(self, get_clipboard: Callable[[], Clipboard | None]) -> None:
        self.get_clipboard = get_clipboard

    def _clipboard(self) -> Clipboard:
        return self.get_clipboard() or DummyClipboard()

    def set_data(self, data: ClipboardData) -> None:
        self._clipboard().set_data(data)

    def set_text(self, text: str) -> None:
        self._clipboard().set_text(text)

    def rotate(self) -> None:
        self._clipboard().rotate()

    def get_data(self) -> ClipboardData:
        return self._clipboard().get_data()