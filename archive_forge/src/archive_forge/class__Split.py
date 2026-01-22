from __future__ import annotations
from abc import ABCMeta, abstractmethod
from enum import Enum
from functools import partial
from typing import TYPE_CHECKING, Callable, Sequence, Union, cast
from prompt_toolkit.application.current import get_app
from prompt_toolkit.cache import SimpleCache
from prompt_toolkit.data_structures import Point
from prompt_toolkit.filters import (
from prompt_toolkit.formatted_text import (
from prompt_toolkit.formatted_text.utils import (
from prompt_toolkit.key_binding import KeyBindingsBase
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType
from prompt_toolkit.utils import get_cwidth, take_using_weights, to_int, to_str
from .controls import (
from .dimension import (
from .margins import Margin
from .mouse_handlers import MouseHandlers
from .screen import _CHAR_CACHE, Screen, WritePosition
from .utils import explode_text_fragments
class _Split(Container):
    """
    The common parts of `VSplit` and `HSplit`.
    """

    def __init__(self, children: Sequence[AnyContainer], window_too_small: Container | None=None, padding: AnyDimension=Dimension.exact(0), padding_char: str | None=None, padding_style: str='', width: AnyDimension=None, height: AnyDimension=None, z_index: int | None=None, modal: bool=False, key_bindings: KeyBindingsBase | None=None, style: str | Callable[[], str]='') -> None:
        self.children = [to_container(c) for c in children]
        self.window_too_small = window_too_small or _window_too_small()
        self.padding = padding
        self.padding_char = padding_char
        self.padding_style = padding_style
        self.width = width
        self.height = height
        self.z_index = z_index
        self.modal = modal
        self.key_bindings = key_bindings
        self.style = style

    def is_modal(self) -> bool:
        return self.modal

    def get_key_bindings(self) -> KeyBindingsBase | None:
        return self.key_bindings

    def get_children(self) -> list[Container]:
        return self.children