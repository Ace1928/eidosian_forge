from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class BinnedTicker(Ticker):
    """ Ticker that aligns ticks exactly at bin boundaries of a scanning color mapper.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    mapper = Instance(ScanningColorMapper, help='\n    A scanning color mapper (e.g. ``EqHistColorMapper``) to use.\n    ')
    num_major_ticks = Either(Int, Auto, default=8, help='\n    The number of major tick positions to show or "auto" to use the\n    number of bins provided by the mapper.\n    ')