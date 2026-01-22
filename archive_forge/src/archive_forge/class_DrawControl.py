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
class DrawControl(Control):
    """DrawControl class.

    Drawing tools for drawing on the map.
    """
    _view_name = Unicode('LeafletDrawControlView').tag(sync=True)
    _model_name = Unicode('LeafletDrawControlModel').tag(sync=True)
    polyline = Dict({'shapeOptions': {}}).tag(sync=True)
    polygon = Dict({'shapeOptions': {}}).tag(sync=True)
    circlemarker = Dict({'shapeOptions': {}}).tag(sync=True)
    circle = Dict().tag(sync=True)
    rectangle = Dict().tag(sync=True)
    marker = Dict().tag(sync=True)
    edit = Bool(True).tag(sync=True)
    remove = Bool(True).tag(sync=True)
    data = List().tag(sync=True)
    last_draw = Dict({'type': 'Feature', 'geometry': None})
    last_action = Unicode()
    _draw_callbacks = Instance(CallbackDispatcher, ())

    def __init__(self, **kwargs):
        super(DrawControl, self).__init__(**kwargs)
        self.on_msg(self._handle_leaflet_event)

    def _handle_leaflet_event(self, _, content, buffers):
        if content.get('event', '').startswith('draw'):
            event, action = content.get('event').split(':')
            self.last_draw = content.get('geo_json')
            self.last_action = action
            self._draw_callbacks(self, action=action, geo_json=self.last_draw)

    def on_draw(self, callback, remove=False):
        """Add a draw event listener.

        Parameters
        ----------
        callback : callable
            Callback function that will be called on draw event.
        remove: boolean
            Whether to remove this callback or not. Defaults to False.
        """
        self._draw_callbacks.register_callback(callback, remove=remove)

    def clear(self):
        """Clear all drawings."""
        self.send({'msg': 'clear'})

    def clear_polylines(self):
        """Clear all polylines."""
        self.send({'msg': 'clear_polylines'})

    def clear_polygons(self):
        """Clear all polygons."""
        self.send({'msg': 'clear_polygons'})

    def clear_circles(self):
        """Clear all circles."""
        self.send({'msg': 'clear_circles'})

    def clear_circle_markers(self):
        """Clear all circle markers."""
        self.send({'msg': 'clear_circle_markers'})

    def clear_rectangles(self):
        """Clear all rectangles."""
        self.send({'msg': 'clear_rectangles'})

    def clear_markers(self):
        """Clear all markers."""
        self.send({'msg': 'clear_markers'})