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