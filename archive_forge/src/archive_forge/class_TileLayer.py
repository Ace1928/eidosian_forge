import copy
import asyncio
import json
import xyzservices
from datetime import date, timedelta
from math import isnan
from branca.colormap import linear, ColorMap
from IPython.display import display
import warnings
from ipywidgets import (
from ipywidgets.widgets.trait_types import InstanceDict
from ipywidgets.embed import embed_minimal_html
from traitlets import (
from ._version import EXTENSION_VERSION
from .projections import projections
class TileLayer(RasterLayer):
    """TileLayer class.

    Tile service layer.

    Attributes
    ----------
    url: string, default "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        Url to the tiles service.
    min_zoom: int, default 0
        The minimum zoom level down to which this layer will be displayed (inclusive).
    max_zoom: int, default 18
        The maximum zoom level up to which this layer will be displayed (inclusive).
    min_native_zoom: int, default None
        Minimum zoom number the tile source has available. If it is specified, the tiles on all zoom levels lower than min_native_zoom will be loaded from min_native_zoom level and auto-scaled.
    max_native_zoom: int, default None
        Maximum zoom number the tile source has available. If it is specified, the tiles on all zoom levels higher than max_native_zoom will be loaded from max_native_zoom level and auto-scaled.
    bounds: list or None, default None
        List of SW and NE location tuples. e.g. [(50, 75), (75, 120)].
    tile_size: int, default 256
        Tile sizes for this tile service.
    attribution: string, default None.
        Tiles service attribution.
    no_wrap: boolean, default False
        Whether the layer is wrapped around the antimeridian.
    tms: boolean, default False
        If true, inverses Y axis numbering for tiles (turn this on for TMS services).
    zoom_offset: int, default 0
        The zoom number used in tile URLs will be offset with this value.
    show_loading: boolean, default False
        Whether to show a spinner when tiles are loading.
    loading: boolean, default False (dynamically updated)
        Whether the tiles are currently loading.
    detect_retina: boolean, default	False
    opacity: float, default 1.0
    visible: boolean, default True
    """
    _view_name = Unicode('LeafletTileLayerView').tag(sync=True)
    _model_name = Unicode('LeafletTileLayerModel').tag(sync=True)
    bottom = Bool(True).tag(sync=True)
    url = Unicode('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').tag(sync=True)
    min_zoom = Int(0).tag(sync=True, o=True)
    max_zoom = Int(18).tag(sync=True, o=True)
    min_native_zoom = Int(default_value=None, allow_none=True).tag(sync=True, o=True)
    max_native_zoom = Int(default_value=None, allow_none=True).tag(sync=True, o=True)
    bounds = List(default_value=None, allow_none=True, help='list of SW and NE location tuples').tag(sync=True, o=True)
    tile_size = Int(256).tag(sync=True, o=True)
    attribution = Unicode(default_value=None, allow_none=True).tag(sync=True, o=True)
    detect_retina = Bool(False).tag(sync=True, o=True)
    no_wrap = Bool(False).tag(sync=True, o=True)
    tms = Bool(False).tag(sync=True, o=True)
    zoom_offset = Int(0).tag(sync=True, o=True)
    show_loading = Bool(False).tag(sync=True)
    loading = Bool(False, read_only=True).tag(sync=True)
    _load_callbacks = Instance(CallbackDispatcher, ())

    def __init__(self, **kwargs):
        super(TileLayer, self).__init__(**kwargs)
        self.on_msg(self._handle_leaflet_event)

    def _handle_leaflet_event(self, _, content, buffers):
        if content.get('event', '') == 'load':
            self._load_callbacks(**content)

    def on_load(self, callback, remove=False):
        """Add a load event listener.

        Parameters
        ----------
        callback : callable
            Callback function that will be called when the tiles have finished loading.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
        self._load_callbacks.register_callback(callback, remove=remove)

    def redraw(self):
        """Force redrawing the tiles.

        This is especially useful when you are sure the server updated the tiles and you
        need to refresh the layer.
        """
        self.send({'msg': 'redraw'})