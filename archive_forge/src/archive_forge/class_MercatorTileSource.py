from __future__ import annotations
import logging # isort:skip
from ..core.properties import (
from ..model import Model
class MercatorTileSource(TileSource):
    """ A base class for Mercator tile services (e.g. ``WMTSTileSource``).

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    _args = ('url', 'tile_size', 'min_zoom', 'max_zoom', 'x_origin_offset', 'y_origin_offset', 'extra_url_vars', 'initial_resolution')
    x_origin_offset = Override(default=20037508.34)
    y_origin_offset = Override(default=20037508.34)
    initial_resolution = Override(default=156543.03392804097)
    snap_to_zoom = Bool(default=False, help='\n    Forces initial extents to snap to the closest larger zoom level.')
    wrap_around = Bool(default=True, help='\n    Enables continuous horizontal panning by wrapping the x-axis based on\n    bounds of map.\n\n    .. note::\n        Axis coordinates are not wrapped. To toggle axis label visibility,\n        use ``plot.axis.visible = False``.\n\n    ')