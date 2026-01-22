import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
@register_mark('bqplot.ScatterGL')
class ScatterGL(Scatter):
    _view_name = Unicode('ScatterGL').tag(sync=True)
    _model_name = Unicode('ScatterGLModel').tag(sync=True)