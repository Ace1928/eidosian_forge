from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class FixedTicker(ContinuousTicker):
    """ Generate ticks at fixed, explicitly supplied locations.

    .. note::
        The ``desired_num_ticks`` property is ignored by this Ticker.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    ticks = Seq(Float, default=[], help='\n    List of major tick locations.\n    ')
    minor_ticks = Seq(Float, default=[], help='\n    List of minor tick locations.\n    ')