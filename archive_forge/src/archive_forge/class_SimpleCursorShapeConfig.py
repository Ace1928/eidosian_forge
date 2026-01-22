from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
class SimpleCursorShapeConfig(CursorShapeConfig):
    """
    Always show the given cursor shape.
    """

    def __init__(self, cursor_shape: CursorShape=CursorShape._NEVER_CHANGE) -> None:
        self.cursor_shape = cursor_shape

    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        return self.cursor_shape