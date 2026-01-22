from __future__ import annotations
import typing
import warnings
from urwid.canvas import CompositeCanvas, SolidCanvas
from urwid.split_repr import remove_defaults
from urwid.util import int_scale
from .constants import (
from .widget_decoration import WidgetDecoration, WidgetError, WidgetWarning
def _get_align(self) -> Literal['left', 'center', 'right'] | tuple[Literal['relative'], int]:
    warnings.warn(f'Method `{self.__class__.__name__}._get_align` is deprecated, please use property `{self.__class__.__name__}.align`', DeprecationWarning, stacklevel=2)
    return self.align