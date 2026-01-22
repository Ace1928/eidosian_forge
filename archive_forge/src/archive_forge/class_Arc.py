from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Arc(XYGlyph, LineGlyph):
    """ Render arcs.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Arc.py'
    _args = ('x', 'y', 'radius', 'start_angle', 'end_angle', 'direction')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the arcs.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the arcs.\n    ')
    radius = DistanceSpec(default=field('radius'), help='\n    Radius of the arc.\n    ')
    start_angle = AngleSpec(default=field('start_angle'), help='\n    The angles to start the arcs, as measured from the horizontal.\n    ')
    end_angle = AngleSpec(default=field('end_angle'), help='\n    The angles to end the arcs, as measured from the horizontal.\n    ')
    direction = Enum(Direction, default='anticlock', help='\n    Which direction to stroke between the start and end angles.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the arcs.\n    ')