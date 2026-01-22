from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class HSpan(LineGlyph):
    """ Horizontal lines of infinite width. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/HSpan.py'
    _args = 'y'
    y = NumberSpec(default=field('y'), help='\n    The y-coordinates of the spans.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the spans.\n    ')