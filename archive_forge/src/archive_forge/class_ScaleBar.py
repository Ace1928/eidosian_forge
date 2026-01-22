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
class ScaleBar(Annotation):
    """ Represents a scale bar annotation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    @error(NON_MATCHING_SCALE_BAR_UNIT)
    def _check_non_matching_scale_bar_unit(self):
        if not self.dimensional.is_known(self.unit):
            return str(self)
    range = Either(Instance(Range), Auto, default='auto', help='\n    The range for which to display the scale.\n\n    This can be either a range reference or ``"auto"``, in which case the\n    scale bar will pick the default x or y range of the frame, depending\n    on the orientation of the scale bar.\n    ')
    unit = String(default='m', help='\n    The unit of the ``range`` property.\n    ')
    dimensional = Instance(Dimensional, default=InstanceDefault(MetricLength), help='\n    Defines the units of measurement.\n    ')
    orientation = Enum(Orientation, help='\n    Whether the scale bar should be oriented horizontally or vertically.\n    ')
    location = Enum(Anchor, default='top_right', help='\n    Location anchor for positioning scale bar.\n    ')
    length_sizing = Enum('adaptive', 'exact', help='\n    Defines how the length of the bar is interpreted.\n\n    This can either be:\n\n    * ``"adaptive"`` - the computed length is fit into a set of ticks provided\n      be the dimensional model. If no ticks are provided, then the behavior\n      is the same as if ``"exact"`` sizing was used\n    * ``"exact"`` - the computed length is used as-is\n\n    ')
    bar_length = NonNegative(Either(Float, Int))(default=0.2, help='\n    The length of the bar, either a fraction of the frame or a number of pixels.\n    ')
    bar_line = Include(ScalarLineProps, prefix='bar', help='\n    The {prop} values for the bar line style.\n    ')
    margin = Int(default=10, help='\n    Amount of margin (in pixels) around the outside of the scale bar.\n    ')
    padding = Int(default=10, help='\n    Amount of padding (in pixels) between the contents of the scale bar\n    and its border.\n    ')
    label = String(default='@{value} @{unit}', help='\n    The label template.\n\n    This can use special variables:\n\n    * ``@{value}`` The current value. Optionally can provide a number\n      formatter with e.g. ``@{value}{%.2f}``.\n    * ``@{unit}`` The unit of measurement.\n\n    ')
    label_text = Include(ScalarTextProps, prefix='label', help='\n    The {prop} values for the label text style.\n    ')
    label_align = Enum(Align, default='center', help="\n    Specifies where to align scale bar's label along the bar.\n\n    This property effective when placing the label above or below\n    a horizontal scale bar, or left or right of a vertical one.\n    ")
    label_location = Enum(Location, default='below', help='\n    Specifies on which side of the scale bar the label will be located.\n    ')
    label_standoff = Int(default=5, help='\n    The distance (in pixels) to separate the label from the scale bar.\n    ')
    title = String(default='', help='\n    The title text to render.\n    ')
    title_text = Include(ScalarTextProps, prefix='title', help='\n    The {prop} values for the title text style.\n    ')
    title_align = Enum(Align, default='center', help="\n    Specifies where to align scale bar's title along the bar.\n\n    This property effective when placing the title above or below\n    a horizontal scale bar, or left or right of a vertical one.\n    ")
    title_location = Enum(Location, default='above', help='\n    Specifies on which side of the legend the title will be located.\n    ')
    title_standoff = Int(default=5, help='\n    The distance (in pixels) to separate the title from the scale bar.\n    ')
    ticker = Instance(Ticker, default=InstanceDefault(FixedTicker, ticks=[]), help='\n    A ticker to use for computing locations of axis components.\n\n    Note that if using the default fixed ticker with no predefined ticks,\n    then the appearance of the scale bar will be just a solid bar with\n    no additional markings.\n    ')
    border_line = Include(ScalarLineProps, prefix='border', help='\n    The {prop} for the scale bar border line style.\n    ')
    background_fill = Include(ScalarFillProps, prefix='background', help='\n    The {prop} for the scale bar background fill style.\n    ')
    background_hatch = Include(ScalarHatchProps, prefix='background', help='\n    The {prop} for the scale bar background hatch style.\n    ')
    bar_line_width = Override(default=2)
    border_line_color = Override(default='#e5e5e5')
    border_line_alpha = Override(default=0.5)
    border_line_width = Override(default=1)
    background_fill_color = Override(default='#ffffff')
    background_fill_alpha = Override(default=0.95)
    label_text_font_size = Override(default='13px')
    label_text_baseline = Override(default='middle')
    title_text_font_size = Override(default='13px')
    title_text_font_style = Override(default='italic')