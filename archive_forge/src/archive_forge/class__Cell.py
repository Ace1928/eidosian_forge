from dataclasses import dataclass, field, replace
from typing import (
from . import box, errors
from ._loop import loop_first_last, loop_last
from ._pick import pick_bool
from ._ratio import ratio_distribute, ratio_reduce
from .align import VerticalAlignMethod
from .jupyter import JupyterMixin
from .measure import Measurement
from .padding import Padding, PaddingDimensions
from .protocol import is_renderable
from .segment import Segment
from .style import Style, StyleType
from .text import Text, TextType
class _Cell(NamedTuple):
    """A single cell in a table."""
    style: StyleType
    'Style to apply to cell.'
    renderable: 'RenderableType'
    'Cell renderable.'
    vertical: VerticalAlignMethod
    'Cell vertical alignment.'