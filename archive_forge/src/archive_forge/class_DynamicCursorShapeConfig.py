from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
class DynamicCursorShapeConfig(CursorShapeConfig):

    def __init__(self, get_cursor_shape_config: Callable[[], AnyCursorShapeConfig]) -> None:
        self.get_cursor_shape_config = get_cursor_shape_config

    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        return to_cursor_shape_config(self.get_cursor_shape_config()).get_cursor_shape(application)