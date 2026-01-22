from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class Quad(LRTBGlyph):
    """ Render axis-aligned quads.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/Quad.py'
    _args = ('left', 'right', 'top', 'bottom')
    left = NumberSpec(default=field('left'), help='\n    The x-coordinates of the left edges.\n    ')
    right = NumberSpec(default=field('right'), help='\n    The x-coordinates of the right edges.\n    ')
    bottom = NumberSpec(default=field('bottom'), help='\n    The y-coordinates of the bottom edges.\n    ')
    top = NumberSpec(default=field('top'), help='\n    The y-coordinates of the top edges.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the quads.\n    ')
    fill_props = Include(FillProps, help='\n    The {prop} values for the quads.\n    ')
    hatch_props = Include(HatchProps, help='\n    The {prop} values for the quads.\n    ')