from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Quadratic(LineGlyph):
    """ Render parabolas.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Quadratic.py'
    _args = ('x0', 'y0', 'x1', 'y1', 'cx', 'cy')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the starting points.\n    ')
    y0 = NumberSpec(default=field('y0'), help='\n    The y-coordinates of the starting points.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the ending points.\n    ')
    y1 = NumberSpec(default=field('y1'), help='\n    The y-coordinates of the ending points.\n    ')
    cx = NumberSpec(default=field('cx'), help='\n    The x-coordinates of the control points.\n    ')
    cy = NumberSpec(default=field('cy'), help='\n    The y-coordinates of the control points.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the parabolas.\n    ')