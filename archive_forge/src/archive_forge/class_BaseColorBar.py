from __future__ import annotations
import logging # isort:skip
from typing import Any
from ...core.enums import (
from ...core.has_props import abstract
from ...core.properties import (
from ...core.property.vectorization import Field
from ...core.property_mixins import (
from ...core.validation import error
from ...core.validation.errors import (
from ...model import Model
from ..formatters import TickFormatter
from ..labeling import LabelingPolicy, NoOverlap
from ..mappers import ColorMapper
from ..ranges import Range
from ..renderers import GlyphRenderer
from ..tickers import FixedTicker, Ticker
from .annotation import Annotation
from .dimensional import Dimensional, MetricLength
@abstract
class BaseColorBar(Annotation):
    """ Abstract base class for color bars.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    location = Either(Enum(Anchor), Tuple(Float, Float), default='top_right', help="\n    The location where the color bar should draw itself. It's either one of\n    ``bokeh.core.enums.Anchor``'s enumerated values, or a ``(x, y)``\n    tuple indicating an absolute location absolute location in screen\n    coordinates (pixels from the bottom-left corner).\n\n    .. warning::\n        If the color bar is placed in a side panel, the location will likely\n        have to be set to `(0,0)`.\n    ")
    orientation = Either(Enum(Orientation), Auto, default='auto', help='\n    Whether the color bar should be oriented vertically or horizontally.\n    ')
    height = Either(Auto, Int, help='\n    The height (in pixels) that the color scale should occupy.\n    ')
    width = Either(Auto, Int, help='\n    The width (in pixels) that the color scale should occupy.\n    ')
    scale_alpha = Float(1.0, help='\n    The alpha with which to render the color scale.\n    ')
    title = Nullable(TextLike, help='\n    The title text to render.\n    ')
    title_props = Include(ScalarTextProps, prefix='title', help='\n    The {prop} values for the title text.\n    ')
    title_text_font_size = Override(default='13px')
    title_text_font_style = Override(default='italic')
    title_standoff = Int(2, help='\n    The distance (in pixels) to separate the title from the color bar.\n    ')
    ticker = Either(Instance(Ticker), Auto, default='auto', help='\n    A Ticker to use for computing locations of axis components.\n    ')
    formatter = Either(Instance(TickFormatter), Auto, default='auto', help='\n    A ``TickFormatter`` to use for formatting the visual appearance of ticks.\n    ')
    major_label_overrides = Dict(Either(Float, String), TextLike, default={}, help='\n    Provide explicit tick label values for specific tick locations that\n    override normal formatting.\n    ')
    major_label_policy = Instance(LabelingPolicy, default=InstanceDefault(NoOverlap), help='\n    Allows to filter out labels, e.g. declutter labels to avoid overlap.\n    ')
    margin = Int(30, help='\n    Amount of margin (in pixels) around the outside of the color bar.\n    ')
    padding = Int(10, help='\n    Amount of padding (in pixels) between the color scale and color bar border.\n    ')
    major_label_props = Include(ScalarTextProps, prefix='major_label', help='\n    The {prop} of the major tick labels.\n    ')
    major_label_text_font_size = Override(default='11px')
    label_standoff = Int(5, help='\n    The distance (in pixels) to separate the tick labels from the color bar.\n    ')
    major_tick_props = Include(ScalarLineProps, prefix='major_tick', help='\n    The {prop} of the major ticks.\n    ')
    major_tick_line_color = Override(default='#ffffff')
    major_tick_in = Int(default=5, help='\n    The distance (in pixels) that major ticks should extend into the\n    main plot area.\n    ')
    major_tick_out = Int(default=0, help='\n    The distance (in pixels) that major ticks should extend out of the\n    main plot area.\n    ')
    minor_tick_props = Include(ScalarLineProps, prefix='minor_tick', help='\n    The {prop} of the minor ticks.\n    ')
    minor_tick_line_color = Override(default=None)
    minor_tick_in = Int(default=0, help='\n    The distance (in pixels) that minor ticks should extend into the\n    main plot area.\n    ')
    minor_tick_out = Int(default=0, help='\n    The distance (in pixels) that major ticks should extend out of the\n    main plot area.\n    ')
    bar_props = Include(ScalarLineProps, prefix='bar', help='\n    The {prop} for the color scale bar outline.\n    ')
    bar_line_color = Override(default=None)
    border_props = Include(ScalarLineProps, prefix='border', help='\n    The {prop} for the color bar border outline.\n    ')
    border_line_color = Override(default=None)
    background_props = Include(ScalarFillProps, prefix='background', help='\n    The {prop} for the color bar background style.\n    ')
    background_fill_color = Override(default='#ffffff')
    background_fill_alpha = Override(default=0.95)