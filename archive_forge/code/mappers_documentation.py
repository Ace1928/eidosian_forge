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
 Maps 3D data arrays of shape ``(ny, nx, nstack)`` to 2D RGBA images
    of shape ``(ny, nx)`` using a palette of length ``nstack``.

    The mapping occurs in two stages. Firstly the RGB values are calculated
    using a weighted sum of the palette colors in the ``nstack`` direction.
    Then the alpha values are calculated using the ``alpha_mapper`` applied to
    the sum of the array in the ``nstack`` direction.

    The RGB values calculated by the ``alpha_mapper`` are ignored by the color
    mapping but are used in any ``ColorBar`` that is displayed.

    