from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
@abstract
class LRTBGlyph(LineGlyph, FillGlyph, HatchGlyph):
    """ Base class for axis-aligned rectangles. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    border_radius = BorderRadius(default=0, help='\n    Allows the box to have rounded corners.\n\n    .. note::\n        This property is experimental and may change at any point.\n    ')