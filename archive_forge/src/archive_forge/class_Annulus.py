from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Annulus(XYGlyph, LineGlyph, FillGlyph, HatchGlyph):
    """ Render annuli.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Annulus.py'
    _args = ('x', 'y', 'inner_radius', 'outer_radius')
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the center of the annuli.\n    ')
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the center of the annuli.\n    ')
    inner_radius = DistanceSpec(default=field('inner_radius'), help='\n    The inner radii of the annuli.\n    ')
    outer_radius = DistanceSpec(default=field('outer_radius'), help='\n    The outer radii of the annuli.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the annuli.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the annuli.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the annuli.\n    ')