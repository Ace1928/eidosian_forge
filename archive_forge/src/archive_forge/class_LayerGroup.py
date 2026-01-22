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
class LayerGroup(Layer):
    """LayerGroup class.

    A group of layers that you can put on the map like other layers.

    Attributes
    ----------
    layers: list, default []
        List of layers to include in the group.
    """
    _view_name = Unicode('LeafletLayerGroupView').tag(sync=True)
    _model_name = Unicode('LeafletLayerGroupModel').tag(sync=True)
    layers = Tuple().tag(trait=Instance(Layer), sync=True, **widget_serialization)
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

    def add_layer(self, layer):
        """Add a new layer to the group.

        .. deprecated :: 0.17.0
           Use add method instead.

        Parameters
        ----------
        layer: layer instance
            The new layer to include in the group.
        """
        warnings.warn('add_layer is deprecated, use add instead', DeprecationWarning)
        self.add(layer)

    def remove_layer(self, rm_layer):
        """Remove a layer from the group.

        .. deprecated :: 0.17.0
           Use remove method instead.

        Parameters
        ----------
        layer: layer instance
            The layer to remove from the group.
        """
        warnings.warn('remove_layer is deprecated, use remove instead', DeprecationWarning)
        self.remove(rm_layer)

    def substitute_layer(self, old, new):
        """Substitute a layer with another one in the group.

        .. deprecated :: 0.17.0
           Use substitute method instead.

        Parameters
        ----------
        old: layer instance
            The layer to remove from the group.
        new: layer instance
            The new layer to include in the group.
        """
        warnings.warn('substitute_layer is deprecated, use substitute instead', DeprecationWarning)
        self.substitute(old, new)

    def clear_layers(self):
        """Remove all layers from the group.

        .. deprecated :: 0.17.0
           Use clear method instead.

        """
        warnings.warn('clear_layers is deprecated, use clear instead', DeprecationWarning)
        self.layers = ()

    def add(self, layer):
        """Add a new layer to the group.

        Parameters
        ----------
        layer: layer instance
            The new layer to include in the group. This can also be an object
            with an ``as_leaflet_layer`` method which generates a compatible
            layer type.
        """
        if isinstance(layer, dict):
            layer = basemap_to_tiles(layer)
        if layer.model_id in self._layer_ids:
            raise LayerException('layer already in layergroup: %r' % layer)
        self.layers = tuple([layer for layer in self.layers] + [layer])

    def remove(self, rm_layer):
        """Remove a layer from the group.

        Parameters
        ----------
        layer: layer instance
            The layer to remove from the group.
        """
        if rm_layer.model_id not in self._layer_ids:
            raise LayerException('layer not on in layergroup: %r' % rm_layer)
        self.layers = tuple([layer for layer in self.layers if layer.model_id != rm_layer.model_id])

    def substitute(self, old, new):
        """Substitute a layer with another one in the group.

        Parameters
        ----------
        old: layer instance
            The layer to remove from the group.
        new: layer instance
            The new layer to include in the group.
        """
        if isinstance(new, dict):
            new = basemap_to_tiles(new)
        if old.model_id not in self._layer_ids:
            raise LayerException('Could not substitute layer: layer not in layergroup.')
        self.layers = tuple([new if layer.model_id == old.model_id else layer for layer in self.layers])

    def clear(self):
        """Remove all layers from the group."""
        self.layers = ()