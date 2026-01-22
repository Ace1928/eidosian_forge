from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.property_aliases import (
from ..core.property_mixins import (
from .glyph import (
from .mappers import ColorMapper, LinearColorMapper, StackColorMapper
class ImageRGBA(ImageBase):
    """ Render images given as RGBA data.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    _args = ('image', 'x', 'y', 'dw', 'dh', 'dilate')
    image = NumberSpec(default=field('image'), help='\n    The arrays of RGBA data for the images.\n    ')