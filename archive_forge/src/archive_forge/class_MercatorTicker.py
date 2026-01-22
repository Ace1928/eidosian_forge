from __future__ import annotations
import logging # isort:skip
from ..core.enums import LatLon
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from .mappers import ScanningColorMapper
class MercatorTicker(BasicTicker):
    """ Generate nice lat/lon ticks form underlying WebMercator coordinates.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimension = Nullable(Enum(LatLon), help='\n    Specify whether to generate ticks for Latitude or Longitude.\n\n    Projected coordinates are not separable, computing Latitude and Longitude\n    tick locations from Web Mercator requires considering coordinates from\n    both dimensions together. Use this property to specify which result should\n    be returned.\n\n    Typically, if the ticker is for an x-axis, then dimension should be\n    ``"lon"`` and if the ticker is for a y-axis, then the dimension\n    should be `"lat"``.\n\n    In order to prevent hard to debug errors, there is no default value for\n    dimension. Using an un-configured ``MercatorTicker`` will result in a\n    validation error and a JavaScript console error.\n    ')

    @error(MISSING_MERCATOR_DIMENSION)
    def _check_missing_dimension(self):
        if self.dimension is None:
            return str(self)