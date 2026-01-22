from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
@runtime_checkable
class WidgetProto(Protocol):
    """Protocol for widget.

    Due to protocol cannot inherit non-protocol bases, define several obligatory Widget methods.
    """

    def sizing(self) -> frozenset[Sizing]:
        ...

    def selectable(self) -> bool:
        ...

    def pack(self, size: tuple[int, int], focus: bool=False) -> tuple[int, int]:
        ...

    @property
    def base_widget(self) -> Widget:
        raise NotImplementedError

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        ...

    def mouse_event(self, size: tuple[int, int], event: str, button: int, col: int, row: int, focus: bool) -> bool | None:
        ...

    def render(self, size: tuple[int, int], focus: bool=False) -> Canvas:
        ...