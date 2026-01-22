from __future__ import annotations
import logging # isort:skip
from ..colors import RGB, Color, ColorLike
from ..core.enums import (
from ..core.has_props import HasProps, abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.property_aliases import GridSpacing, Pixels, TracksSizing
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import (
from ..model import Model
from .ui.panes import Pane
from .ui.tooltips import Tooltip
from .ui.ui_element import UIElement
@error(MIN_PREFERRED_MAX_WIDTH)
def _check_min_preferred_max_width(self):
    min_width = self.min_width if self.min_width is not None else 0
    width = self.width if self.width is not None and (self.sizing_mode == 'fixed' or self.width_policy == 'fixed') else min_width
    max_width = self.max_width if self.max_width is not None else width
    if not min_width <= width <= max_width:
        return str(self)