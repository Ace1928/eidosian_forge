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
class DynamicContainer(Container):
    """
    Container class that dynamically returns any Container.

    :param get_container: Callable that returns a :class:`.Container` instance
        or any widget with a ``__pt_container__`` method.
    """

    def __init__(self, get_container: Callable[[], AnyContainer]) -> None:
        self.get_container = get_container

    def _get_container(self) -> Container:
        """
        Return the current container object.

        We call `to_container`, because `get_container` can also return a
        widget with a ``__pt_container__`` method.
        """
        obj = self.get_container()
        return to_container(obj)

    def reset(self) -> None:
        self._get_container().reset()

    def preferred_width(self, max_available_width: int) -> Dimension:
        return self._get_container().preferred_width(max_available_width)

    def preferred_height(self, width: int, max_available_height: int) -> Dimension:
        return self._get_container().preferred_height(width, max_available_height)

    def write_to_screen(self, screen: Screen, mouse_handlers: MouseHandlers, write_position: WritePosition, parent_style: str, erase_bg: bool, z_index: int | None) -> None:
        self._get_container().write_to_screen(screen, mouse_handlers, write_position, parent_style, erase_bg, z_index)

    def is_modal(self) -> bool:
        return False

    def get_key_bindings(self) -> KeyBindingsBase | None:
        return None

    def get_children(self) -> list[Container]:
        return [self._get_container()]