from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Bezier(LineGlyph):
    """ Render Bezier curves.

    For more information consult the `Wikipedia article for Bezier curve`_.

    .. _Wikipedia article for Bezier curve: http://en.wikipedia.org/wiki/Bezier_curve

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Bezier.py'
    _args = ('x0', 'y0', 'x1', 'y1', 'cx0', 'cy0', 'cx1', 'cy1')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the starting points.\n    ')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the starting points.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the ending points.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the ending points.\n    ')
    cx0 = NumberSpec(default=field('cx0'), help='\n    The x-coordinates of first control points.\n    ')
    cy0 = NumberSpec(default=field('cy0'), help='\n    The y-coordinates of first control points.\n    ')
    cx1 = NumberSpec(default=field('cx1'), help='\n    The x-coordinates of second control points.\n    ')
    cy1 = NumberSpec(default=field('cy1'), help='\n    The y-coordinates of second control points.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the Bezier curves.\n    ')