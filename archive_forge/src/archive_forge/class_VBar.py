from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class VBar(LRTBGlyph):
    """ Render vertical bars, given a center coordinate, width and (top, bottom) coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VBar.py'
    _args = ('x', 'width', 'top', 'bottom')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the centers of the vertical bars.\n    ')
    width = DistanceSpec(default=1, help='\n    The widths of the vertical bars.\n    ')
    bottom = NumberSpec(default=0, help='\n    The y-coordinates of the bottom edges.\n    ')
    top = NumberSpec(default=field('top'), help='\n    The y-coordinates of the top edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the vertical bars.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the vertical bars.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the vertical bars.\n    ')