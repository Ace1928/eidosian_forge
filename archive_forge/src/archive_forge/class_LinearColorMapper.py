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
class LinearColorMapper(ContinuousColorMapper):
    """ Map numbers in a range [*low*, *high*] linearly into a sequence of
    colors (a palette).

    For example, if the range is [0, 99] and the palette is
    ``['red', 'green', 'blue']``, the values would be mapped as follows::

             x < 0  : 'red'     # values < low are clamped
        0 <= x < 33 : 'red'
       33 <= x < 66 : 'green'
       66 <= x < 99 : 'blue'
       99 <= x      : 'blue'    # values > high are clamped

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)