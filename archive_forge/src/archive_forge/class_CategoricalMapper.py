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
class CategoricalMapper(Mapper):
    """ Base class for mappers that map categorical factors to other values.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    factors = FactorSeq(help='\n    A sequence of factors / categories that map to the some target range. For\n    example the following color mapper:\n\n    .. code-block:: python\n\n        mapper = CategoricalColorMapper(palette=["red", "blue"], factors=["foo", "bar"])\n\n    will map the factor ``"foo"`` to red and the factor ``"bar"`` to blue.\n    ')
    start = Int(default=0, help='\n    A start index to "slice" data factors with before mapping.\n\n    For example, if the data to color map consists of 2-level factors such\n    as ``["2016", "sales"]`` and ``["2016", "marketing"]``, then setting\n    ``start=1`` will perform color mapping only based on the second sub-factor\n    (i.e. in this case based on the department ``"sales"`` or ``"marketing"``)\n    ')
    end = Nullable(Int, help='\n    A start index to "slice" data factors with before mapping.\n\n    For example, if the data to color map consists of 2-level factors such\n    as ``["2016", "sales"]`` and ``["2017", "marketing"]``, then setting\n    ``end=1`` will perform color mapping only based on the first sub-factor\n    (i.e. in this case based on the year ``"2016"`` or ``"2017"``)\n\n    If ``None`` then all sub-factors from ``start`` to the end of the\n    factor will be used for color mapping.\n    ')