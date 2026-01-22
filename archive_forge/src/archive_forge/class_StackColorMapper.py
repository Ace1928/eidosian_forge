from __future__ import annotations
import logging # isort:skip
from .. import palettes
from ..core.enums import Palette
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import WEIGHTED_STACK_COLOR_MAPPER_LABEL_LENGTH_MISMATCH
from ..core.validation.warnings import PALETTE_LENGTH_FACTORS_MISMATCH
from .transforms import Transform
@abstract
class StackColorMapper(ColorMapper):
    """ Abstract base class for color mappers that operate on ``ImageStack``
    glyphs.

    These map 3D data arrays of shape ``(ny, nx, nstack)`` to 2D RGBA images
    of shape ``(ny, nx)``.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)