from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
@abstract
class ContinuousTicker(Ticker):
    """ A base class for non-categorical ticker types.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    num_minor_ticks = Int(5, help='\n    The number of minor tick positions to generate between\n    adjacent major tick values.\n    ')
    desired_num_ticks = Int(6, help='\n    A desired target number of major tick positions to generate across\n    the plot range.\n\n    .. note:\n        This value is a suggestion, and ticker subclasses may ignore\n        it entirely, or use it only as an ideal goal to approach as well\n        as can be, in the context of a specific ticking strategy.\n    ')