from __future__ import annotations
import typing
from urwid.canvas import CanvasCombine, CompositeCanvas, SolidCanvas
from urwid.util import get_encoding_mode
from .constants import BAR_SYMBOLS, Sizing
from .text import Text
from .widget import Widget, WidgetError, WidgetMeta, nocache_widget_render, nocache_widget_render_instance
def use_smoothed(self) -> bool:
    return self.satt and get_encoding_mode() == 'utf8'