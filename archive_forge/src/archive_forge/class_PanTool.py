from __future__ import annotations
import logging # isort:skip
import difflib
import typing as tp
from math import nan
from typing import Literal
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property.struct import Optional
from ..core.validation import error
from ..core.validation.errors import NO_RANGE_TOOL_RANGES
from ..model import Model
from ..util.strings import nice_join
from .annotations import BoxAnnotation, PolyAnnotation, Span
from .callbacks import Callback
from .dom import Template
from .glyphs import (
from .nodes import Node
from .ranges import Range
from .renderers import DataRenderer, GlyphRenderer
from .ui import UIElement
class PanTool(Drag):
    """ *toolbar icon*: |pan_icon|

    The pan tool allows the user to pan a Plot by left-dragging a mouse, or on
    touch devices by dragging a finger or stylus, across the plot region.

    The pan tool also activates the border regions of a Plot for "single axis"
    panning. For instance, dragging in the vertical border or axis will effect
    a pan in the vertical direction only, with horizontal dimension kept fixed.

    .. |pan_icon| image:: /_images/icons/Pan.png
        :height: 24px
        :alt: Icon of four arrows meeting in a plus shape representing the pan tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimensions = Enum(Dimensions, default='both', help='\n    Which dimensions the pan tool is constrained to act in. By default\n    the pan tool will pan in any dimension, but can be configured to only\n    pan horizontally across the width of the plot, or vertically across the\n    height of the plot.\n    ')