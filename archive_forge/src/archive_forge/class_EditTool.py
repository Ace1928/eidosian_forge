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
@abstract
class EditTool(GestureTool):
    """ A base class for all interactive draw tool types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    default_overrides = Dict(String, Any, default={}, help='\n    Padding values overriding ``ColumnarDataSource.default_values``.\n\n    Defines values to insert into non-coordinate columns when a new glyph is\n    inserted into the ``ColumnDataSource`` columns, e.g. when a circle glyph\n    defines ``"x"``, ``"y"`` and ``"color"`` columns, adding a new point will\n    add the x and y-coordinates to ``"x"`` and ``"y"`` columns and the color\n    column will be filled with the defined default value.\n    ')
    empty_value = Either(Bool, Int, Float, Date, Datetime, Color, String, default=0, help='\n    The "last resort" padding value.\n\n    This is used the same as ``default_values``, when the tool was unable\n    to figure out a default value otherwise. The tool will try the following\n    alternatives in order:\n\n    1. ``EditTool.default_overrides``\n    2. ``ColumnarDataSource.default_values``\n    3. ``ColumnarDataSource``\'s inferred default values\n    4. ``EditTool.empty_value``\n    ')