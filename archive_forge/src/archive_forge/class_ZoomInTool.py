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
class ZoomInTool(ZoomBaseTool):
    """ *toolbar icon*: |zoom_in_icon|

    The zoom-in tool allows users to click a button to zoom in
    by a fixed amount.

    .. |zoom_in_icon| image:: /_images/icons/ZoomIn.png
        :height: 24px
        :alt: Icon of a plus sign next to a magnifying glass representing the zoom-in tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)