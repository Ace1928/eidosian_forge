from __future__ import annotations
import typing
import warnings
from urwid.canvas import CanvasOverlay, CompositeCanvas
from urwid.split_repr import remove_defaults
from .constants import (
from .container import WidgetContainerListContentsMixin, WidgetContainerMixin
from .filler import calculate_top_bottom_filler
from .padding import calculate_left_right_padding
from .widget import Widget, WidgetError, WidgetWarning
class OverlayOptions(typing.NamedTuple):
    align: Align | Literal[WHSettings.RELATIVE]
    align_amount: int | None
    width_type: WHSettings
    width_amount: int | None
    min_width: int | None
    left: int
    right: int
    valign_type: VAlign | Literal[WHSettings.RELATIVE]
    valign_amount: int | None
    height_type: WHSettings
    height_amount: int | None
    min_height: int | None
    top: int
    bottom: int