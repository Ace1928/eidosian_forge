from __future__ import annotations
import typing
from .columns import Columns
from .constants import BOX_SYMBOLS, Align, WHSettings
from .divider import Divider
from .pile import Pile
from .solid_fill import SolidFill
from .text import Text
from .widget_decoration import WidgetDecoration, delegate_to_widget_mixin
@property
def _w(self) -> Pile:
    return self._wrapped_widget