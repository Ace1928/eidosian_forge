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
class LocalTileLayer(TileLayer):
    """LocalTileLayer class.

    Custom tile layer using local tile files.

    Attributes
    ----------
    path: string, default ""
        Path to your local tiles. In the classic Jupyter Notebook, the path is relative to
        the Notebook you are working on. In JupyterLab, the path is relative to the server
        (where you started JupyterLab) and you need to prefix the path with “files/”.
    """
    _view_name = Unicode('LeafletLocalTileLayerView').tag(sync=True)
    _model_name = Unicode('LeafletLocalTileLayerModel').tag(sync=True)
    path = Unicode('').tag(sync=True)