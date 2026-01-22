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
class PolyDrawTool(PolyTool, Drag, Tap):
    """ *toolbar icon*: |poly_draw_icon|

    The PolyDrawTool allows drawing, selecting and deleting ``Patches`` and
    ``MultiLine`` glyphs on one or more renderers by editing the underlying
    ``ColumnDataSource`` data. Like other drawing tools, the renderers that
    are to be edited must be supplied explicitly.

    The tool will modify the columns on the data source corresponding to the
    ``xs`` and ``ys`` values of the glyph. Any additional columns in the data
    source will be padded with the declared ``empty_value``, when adding a new
    point.

    If a ``vertex_renderer`` with an point-like glyph is supplied then the
    ``PolyDrawTool`` will use it to display the vertices of the multi-lines or
    patches on all supplied renderers. This also enables the ability to snap
    to existing vertices while drawing.

    The supported actions include:

    * Add patch or multi-line: press to add the first vertex, then use tap
      to add each subsequent vertex, to finalize the draw action press to
      insert the final vertex or press the ESC key.

    * Move patch or multi-line: Tap and drag an existing patch/multi-line, the
      point will be dropped once you let go of the mouse button.

    * Delete patch or multi-line: Tap a patch/multi-line to select it then
      press BACKSPACE key while the mouse is within the plot area.

    .. |poly_draw_icon| image:: /_images/icons/PolyDraw.png
        :height: 24px
        :alt: Icon of a solid line trapezoid with an arrow pointing at the lower right representing the polygon-draw tool in the toolbar.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    renderers = List(GlyphRendererOf(MultiLine, Patches), help='\n    A list of renderers corresponding to glyphs that may be edited.\n    ')
    drag = Bool(default=True, help='\n    Enables dragging of existing patches and multi-lines on pan events.\n    ')
    num_objects = Int(default=0, help='\n    Defines a limit on the number of patches or multi-lines that can be drawn.\n    By default there is no limit on the number of objects, but if enabled the\n    oldest drawn patch or multi-line will be dropped to make space for the new\n    patch or multi-line.\n    ')