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
class MagnifyingGlass(RasterLayer):
    """MagnifyingGlass class.

    Attributes
    ----------
    radius: int, default 100
        The radius of the magnifying glass, in pixels.
    zoom_offset: int, default 3
        The zoom level offset between the main map zoom and the magnifying glass.
    fixed_zoom: int, default -1
        If different than -1, defines a fixed zoom level to always use in the magnifying glass,
        ignoring the main map zoom and the zoomOffet value.
    fixed_position: boolean, default False
        If True, the magnifying glass will stay at the same position on the map,
        not following the mouse cursor.
    lat_lng: list, default [0, 0]
        The initial position of the magnifying glass, both on the main map and as the center
        of the magnified view. If fixed_position is True, it will always keep this position.
    layers: list, default []
        Set of layers to display in the magnified view.
        These layers shouldn't be already added to a map instance.
    """
    _view_name = Unicode('LeafletMagnifyingGlassView').tag(sync=True)
    _model_name = Unicode('LeafletMagnifyingGlassModel').tag(sync=True)
    radius = Int(100).tag(sync=True, o=True)
    zoom_offset = Int(3).tag(sync=True, o=True)
    fixed_zoom = Int(-1).tag(sync=True, o=True)
    fixed_position = Bool(False).tag(sync=True, o=True)
    lat_lng = List(def_loc).tag(sync=True, o=True)
    layers = Tuple().tag(trait=Instance(Layer), sync=True, o=True, **widget_serialization)
    _layer_ids = List()

    @validate('layers')
    def _validate_layers(self, proposal):
        """Validate layers list.

        Makes sure only one instance of any given layer can exist in the
        layers list.
        """
        self._layer_ids = [layer.model_id for layer in proposal.value]
        if len(set(self._layer_ids)) != len(self._layer_ids):
            raise LayerException('duplicate layer detected, only use each layer once')
        return proposal.value