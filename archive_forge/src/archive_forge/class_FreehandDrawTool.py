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
class FreehandDrawTool(EditTool, Drag, Tap):
    """ *toolbar icon*: |freehand_draw_icon|

    Allows freehand drawing of ``Patches`` and ``MultiLine`` glyphs. The glyph
    to draw may be defined via the ``renderers`` property.

    The tool will modify the columns on the data source corresponding to the
    ``xs`` and ``ys`` values of the glyph. Any additional columns in the data
    source will be padded with the declared ``empty_value``, when adding a new
    point.

    The supported actions include:

    * Draw vertices: Click and drag to draw a line

    * Delete patch/multi-line: Tap a patch/multi-line to select it then press
      BACKSPACE key while the mouse is within the plot area.

    .. |freehand_draw_icon| image:: /_images/icons/FreehandDraw.png
        :height: 24px
        :alt: Icon of a pen drawing a wavy line representing the freehand-draw tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = List(GlyphRendererOf(MultiLine, Patches), help='\n    A list of renderers corresponding to glyphs that may be edited.\n    ')
    num_objects = Int(default=0, help='\n    Defines a limit on the number of patches or multi-lines that can be drawn.\n    By default there is no limit on the number of objects, but if enabled the\n    oldest drawn patch or multi-line will be overwritten when the limit is\n    reached.\n    ')