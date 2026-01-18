from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
def to_cursor_shape_config(value: AnyCursorShapeConfig) -> CursorShapeConfig:
    """
    Take a `CursorShape` instance or `CursorShapeConfig` and turn it into a
    `CursorShapeConfig`.
    """
    if value is None:
        return SimpleCursorShapeConfig()
    if isinstance(value, CursorShape):
        return SimpleCursorShapeConfig(value)
    return value