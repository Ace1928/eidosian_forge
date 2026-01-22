from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class VSpan(LineGlyph):
    """ Vertical lines of infinite height. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    __example__ = 'examples/reference/models/VSpan.py'
    _args = 'x'
    x = NumberSpec(default=field('x'), help='\n    The x-coordinates of the spans.\n    ')
    line_props = Include(LineProps, help='\n    The {prop} values for the spans.\n    ')