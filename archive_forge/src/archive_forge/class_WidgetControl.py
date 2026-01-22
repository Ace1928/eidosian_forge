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
class WidgetControl(Control):
    """WidgetControl class, with Control as parent class.

    A control that contains any DOMWidget instance.

    Attributes
    ----------
    widget: DOMWidget
        The widget to put inside of the control. It can be any widget, even coming from
        a third-party library like bqplot.
    """
    _view_name = Unicode('LeafletWidgetControlView').tag(sync=True)
    _model_name = Unicode('LeafletWidgetControlModel').tag(sync=True)
    widget = Instance(DOMWidget).tag(sync=True, **widget_serialization)
    max_width = Int(default_value=None, allow_none=True).tag(sync=True)
    min_width = Int(default_value=None, allow_none=True).tag(sync=True)
    max_height = Int(default_value=None, allow_none=True).tag(sync=True)
    min_height = Int(default_value=None, allow_none=True).tag(sync=True)
    transparent_bg = Bool(False).tag(sync=True, o=True)