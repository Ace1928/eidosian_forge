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
class ZoomControl(Control):
    """ZoomControl class, with Control as parent class.

    A control which contains buttons for zooming in/out the Map.

    Attributes
    ----------
    zoom_in_text: str, default '+'
        Text to put in the zoom-in button.
    zoom_in_title: str, default 'Zoom in'
        Title to put in the zoom-in button, this is shown when the mouse
        is over the button.
    zoom_out_text: str, default '-'
        Text to put in the zoom-out button.
    zoom_out_title: str, default 'Zoom out'
        Title to put in the zoom-out button, this is shown when the mouse
        is over the button.
    """
    _view_name = Unicode('LeafletZoomControlView').tag(sync=True)
    _model_name = Unicode('LeafletZoomControlModel').tag(sync=True)
    zoom_in_text = Unicode('+').tag(sync=True, o=True)
    zoom_in_title = Unicode('Zoom in').tag(sync=True, o=True)
    zoom_out_text = Unicode('-').tag(sync=True, o=True)
    zoom_out_title = Unicode('Zoom out').tag(sync=True, o=True)