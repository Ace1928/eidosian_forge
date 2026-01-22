from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas, SolidCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError, WidgetWarning
def _set_width(self, width: Literal['clip', 'pack'] | int | tuple[Literal['relative'], int]) -> None:
    warnings.warn(f'Method `{self.__class__.__name__}._set_width` is deprecated, please use property `{self.__class__.__name__}.width`', DeprecationWarning, stacklevel=2)
    self.width = width