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
class LogColorMapper(ContinuousColorMapper):
    """ Map numbers in a range [*low*, *high*] into a sequence of colors
    (a palette) on a natural logarithm scale.

    For example, if the range is [0, 25] and the palette is
    ``['red', 'green', 'blue']``, the values would be mapped as follows::

                x < 0     : 'red'     # values < low are clamped
       0     <= x < 2.72  : 'red'     # math.e ** 1
       2.72  <= x < 7.39  : 'green'   # math.e ** 2
       7.39  <= x < 20.09 : 'blue'    # math.e ** 3
       20.09 <= x         : 'blue'    # values > high are clamped

    .. warning::
        The ``LogColorMapper`` only works for images with scalar values that are
        non-negative.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)