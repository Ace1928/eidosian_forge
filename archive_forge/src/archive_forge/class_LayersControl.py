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
class LayersControl(Control):
    """LayersControl class, with Control as parent class.

    A control which allows hiding/showing different layers on the Map.
    Attributes
    ----------------------
    collapsed: bool, default True
        Set whether control should be open or closed by default
    """
    _view_name = Unicode('LeafletLayersControlView').tag(sync=True)
    _model_name = Unicode('LeafletLayersControlModel').tag(sync=True)
    collapsed = Bool(True).tag(sync=True, o=True)