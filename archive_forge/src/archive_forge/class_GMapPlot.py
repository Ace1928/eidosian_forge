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
class GMapPlot(MapPlot):
    """ A Bokeh Plot with a `Google Map`_ displayed underneath.

    Data placed on this plot should be specified in decimal lat/lon coordinates
    e.g. ``(37.123, -122.404)``. It will be automatically converted into the
    web mercator projection to display properly over google maps tiles.

    The ``api_key`` property must be configured with a Google API Key in order
    for ``GMapPlot`` to function. The key will be stored in the Bokeh Document
    JSON.

    Note that Google Maps exert explicit control over aspect ratios at all
    times, which imposes some limitations on ``GMapPlot``:

    * Only ``Range1d`` ranges are supported. Attempting to use other range
      types will result in an error.

    * Usage of ``BoxZoomTool`` is incompatible with ``GMapPlot``. Adding a
      ``BoxZoomTool`` will have no effect.

    .. _Google Map: https://www.google.com/maps/

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @error(REQUIRED_RANGE)
    def _check_required_range(self):
        pass

    @warning(MISSING_RENDERERS)
    def _check_missing_renderers(self):
        pass

    @error(MISSING_GOOGLE_API_KEY)
    def _check_missing_google_api_key(self):
        if self.api_key is None:
            return str(self)
    map_options = Instance(GMapOptions, help='\n    Options for displaying the plot.\n    ')
    border_fill_color = Override(default='#ffffff')
    background_fill_alpha = Override(default=0.0)
    api_key = Required(Bytes, help='\n    Google Maps API requires an API key. See https://developers.google.com/maps/documentation/javascript/get-api-key\n    for more information on how to obtain your own.\n    ').accepts(String, lambda val: val.encode('utf-8'))
    api_version = String(default='weekly', help='\n    The version of Google Maps API to use. See https://developers.google.com/maps/documentation/javascript/versions\n    for more information.\n\n    .. note::\n        Changing this value may result in broken map rendering.\n\n    ')
    x_range = Override(default=InstanceDefault(Range1d))
    y_range = Override(default=InstanceDefault(Range1d))