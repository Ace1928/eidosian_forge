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
class Toolbar(UIElement):
    """ Collect tools to display for a single plot.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    logo = Nullable(Enum('normal', 'grey'), default='normal', help='\n    What version of the Bokeh logo to display on the toolbar. If\n    set to None, no logo will be displayed.\n    ')
    autohide = Bool(default=False, help='\n    Whether the toolbar will be hidden by default. Default: False.\n    If True, hides toolbar when cursor is not in canvas.\n    ')
    tools = List(Either(Instance(Tool), Instance(ToolProxy)), help='\n    A list of tools to add to the plot.\n    ')
    active_drag: Literal['auto'] | Drag | None = Either(Null, Auto, Instance(Drag), default='auto', help='\n    Specify a drag tool to be active when the plot is displayed.\n    ')
    active_inspect: Literal['auto'] | InspectTool | tp.Sequence[InspectTool] | None = Either(Null, Auto, Instance(InspectTool), Seq(Instance(InspectTool)), default='auto', help='\n    Specify an inspection tool or sequence of inspection tools to be active when\n    the plot is displayed.\n    ')
    active_scroll: Literal['auto'] | Scroll | None = Either(Null, Auto, Instance(Scroll), default='auto', help='\n    Specify a scroll/pinch tool to be active when the plot is displayed.\n    ')
    active_tap: Literal['auto'] | Tap | None = Either(Null, Auto, Instance(Tap), default='auto', help='\n    Specify a tap/click tool to be active when the plot is displayed.\n    ')
    active_multi: Literal['auto'] | GestureTool | None = Either(Null, Auto, Instance(GestureTool), default='auto', help='\n    Specify an active multi-gesture tool, for instance an edit tool or a range\n    tool.\n\n    Note that activating a multi-gesture tool will deactivate any other gesture\n    tools as appropriate. For example, if a pan tool is set as the active drag,\n    and this property is set to a ``BoxEditTool`` instance, the pan tool will\n    be deactivated (i.e. the multi-gesture tool will take precedence).\n    ')