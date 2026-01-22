from __future__ import annotations
import logging # isort:skip
from ..core.enums import MapType
from ..core.has_props import abstract
from ..core.properties import (
from ..core.validation import error, warning
from ..core.validation.errors import (
from ..core.validation.warnings import MISSING_RENDERERS
from ..model import Model
from ..models.ranges import Range1d
from .plots import Plot
class GMapOptions(MapOptions):
    """ Options for ``GMapPlot`` objects.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    map_type = Enum(MapType, default='roadmap', help='\n    The `map type`_ to use for the ``GMapPlot``.\n\n    .. _map type: https://developers.google.com/maps/documentation/javascript/reference#MapTypeId\n\n    ')
    scale_control = Bool(default=False, help='\n    Whether the Google map should display its distance scale control.\n    ')
    styles = Nullable(JSON, default=None, help='\n    A JSON array of `map styles`_ to use for the ``GMapPlot``. Many example styles can\n    `be found here`_.\n\n    .. _map styles: https://developers.google.com/maps/documentation/javascript/reference#MapTypeStyle\n    .. _be found here: https://snazzymaps.com\n\n    ')
    tilt = Int(default=45, help="\n    `Tilt`_ angle of the map. The only allowed values are 0 and 45.\n    Only has an effect on 'satellite' and 'hybrid' map types.\n    A value of 0 causes the map to always use a 0 degree overhead view.\n    A value of 45 causes the tilt angle to switch to 45 imagery if available.\n\n    .. _Tilt: https://developers.google.com/maps/documentation/javascript/reference/3/map#MapOptions.tilt\n\n    ")