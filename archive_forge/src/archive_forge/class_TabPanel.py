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
class TabPanel(Model):
    """ A single-widget container with title bar and controls.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    title = String(default='', help='\n    The text title of the panel.\n    ')
    tooltip = Nullable(Instance(Tooltip), default=None, help="\n    A tooltip with plain text or rich HTML contents, providing general help or\n    description of a widget's or component's function.\n    ")
    child = Instance(UIElement, help='\n    The child widget. If you need more children, use a layout widget, e.g. a ``Column``.\n    ')
    closable = Bool(False, help='\n    Whether this panel is closable or not. If True, an "x" button will appear.\n\n    Closing a panel is equivalent to removing it from its parent container (e.g. tabs).\n    ')
    disabled = Bool(False, help='\n    Whether the widget is responsive to UI events.\n    ')