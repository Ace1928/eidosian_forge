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
def _divide_heights(self, write_position: WritePosition) -> list[int] | None:
    """
        Return the heights for all rows.
        Or None when there is not enough space.
        """
    if not self.children:
        return []
    width = write_position.width
    height = write_position.height
    dimensions = [c.preferred_height(width, height) for c in self._all_children]
    sum_dimensions = sum_layout_dimensions(dimensions)
    if sum_dimensions.min > height:
        return None
    sizes = [d.min for d in dimensions]
    child_generator = take_using_weights(items=list(range(len(dimensions))), weights=[d.weight for d in dimensions])
    i = next(child_generator)
    preferred_stop = min(height, sum_dimensions.preferred)
    preferred_dimensions = [d.preferred for d in dimensions]
    while sum(sizes) < preferred_stop:
        if sizes[i] < preferred_dimensions[i]:
            sizes[i] += 1
        i = next(child_generator)
    if not get_app().is_done:
        max_stop = min(height, sum_dimensions.max)
        max_dimensions = [d.max for d in dimensions]
        while sum(sizes) < max_stop:
            if sizes[i] < max_dimensions[i]:
                sizes[i] += 1
            i = next(child_generator)
    return sizes