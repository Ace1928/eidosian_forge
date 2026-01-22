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
class VideoOverlay(RasterLayer):
    """VideoOverlay class.

    Video layer from a local or remote video file.

    Attributes
    ----------
    url: string, default ""
        Url to the local or remote video file.
    bounds: list, default [0., 0]
        SW and NE corners of the video.
    attribution: string, default ""
        Video attribution.
    """
    _view_name = Unicode('LeafletVideoOverlayView').tag(sync=True)
    _model_name = Unicode('LeafletVideoOverlayModel').tag(sync=True)
    url = Unicode().tag(sync=True)
    bounds = List([def_loc, def_loc], help='SW and NE corners of the image').tag(sync=True)
    attribution = Unicode().tag(sync=True, o=True)