from __future__ import annotations
import logging # isort:skip
from ..core.enums import (
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error
from ..core.validation.errors import MISSING_MERCATOR_DIMENSION
from ..model import Model
from ..util.deprecation import deprecated
from ..util.strings import format_docstring
from ..util.warnings import warn
from .tickers import Ticker
class MercatorTickFormatter(BasicTickFormatter):
    """ A ``TickFormatter`` for values in WebMercator units.

    Some map plot types internally use WebMercator to describe coordinates,
    plot bounds, etc. These units are not very human-friendly. This tick
    formatter will convert WebMercator units into Latitude and Longitude
    for display on axes.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    dimension = Nullable(Enum(LatLon), help='\n    Specify whether to format ticks for Latitude or Longitude.\n\n    Projected coordinates are not separable, computing Latitude and Longitude\n    tick labels from Web Mercator requires considering coordinates from both\n    dimensions together. Use this property to specify which result should be\n    used for display.\n\n    Typically, if the formatter is for an x-axis, then dimension should be\n    ``"lon"`` and if the formatter is for a y-axis, then the dimension\n    should be `"lat"``.\n\n    In order to prevent hard to debug errors, there is no default value for\n    dimension. Using an un-configured ``MercatorTickFormatter`` will result in\n    a validation error and a JavaScript console error.\n    ')

    @error(MISSING_MERCATOR_DIMENSION)
    def _check_missing_dimension(self):
        if self.dimension is None:
            return str(self)