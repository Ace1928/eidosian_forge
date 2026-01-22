from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Union
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.key_binding.vi_state import InputMode
class ModalCursorShapeConfig(CursorShapeConfig):
    """
    Show cursor shape according to the current input mode.
    """

    def get_cursor_shape(self, application: Application[Any]) -> CursorShape:
        if application.editing_mode == EditingMode.VI:
            if application.vi_state.input_mode == InputMode.INSERT:
                return CursorShape.BEAM
            if application.vi_state.input_mode == InputMode.REPLACE:
                return CursorShape.UNDERLINE
        return CursorShape.BLOCK