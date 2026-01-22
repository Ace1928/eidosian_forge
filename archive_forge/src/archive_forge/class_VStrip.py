from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class VStrip(LineGlyph, FillGlyph, HatchGlyph):
    """ Vertical strips of infinite height. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VStrip.py'
    _args = ('x0', 'x1')
    x0 = NumberSpec(default=field('x0'), help='\n    The x-coordinates of the coordinates of one side of the strips.\n    ')
    x1 = NumberSpec(default=field('x1'), help='\n    The x-coordinates of the coordinates of the other side of the strips.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the strips.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the strips.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the strips.\n    ')