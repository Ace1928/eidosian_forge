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
class ScaleControl(Control):
    """ScaleControl class, with Control as parent class.

    A control which shows the Map scale.

    Attributes
    ----------
    max_width: int, default 100
        Max width of the control, in pixels.
    metric: bool, default True
        Whether to show metric units.
    imperial: bool, default True
        Whether to show imperial units.
    """
    _view_name = Unicode('LeafletScaleControlView').tag(sync=True)
    _model_name = Unicode('LeafletScaleControlModel').tag(sync=True)
    max_width = Int(100).tag(sync=True, o=True)
    metric = Bool(True).tag(sync=True, o=True)
    imperial = Bool(True).tag(sync=True, o=True)
    update_when_idle = Bool(False).tag(sync=True, o=True)