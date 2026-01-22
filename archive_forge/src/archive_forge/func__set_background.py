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
def _set_background(self, color: ColorLike) -> None:
    """ Background color of the component. """
    if isinstance(color, Color):
        color = color.to_css()
    elif isinstance(color, tuple):
        color = RGB.from_tuple(color).to_css()
    if isinstance(self.styles, dict):
        self.styles['background-color'] = color
    else:
        self.styles.background_color = color