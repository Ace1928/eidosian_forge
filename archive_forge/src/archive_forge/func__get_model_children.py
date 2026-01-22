from __future__ import annotations
import datetime as dt
import uuid
from functools import partial
from types import FunctionType, MethodType
from typing import (
import numpy as np
import param
from bokeh.model import Model
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from bokeh.models.widgets.tables import (
from bokeh.util.serialization import convert_datetime_array
from pyviz_comms import JupyterComm
from ..depends import transform_reference
from ..io.resources import CDN_DIST, CSS_URLS
from ..io.state import state
from ..reactive import Reactive, ReactiveData
from ..util import (
from ..util.warnings import warn
from .base import Widget
from .button import Button
from .input import TextInput
def _get_model_children(self, panels, doc, root, parent, comm=None):
    ref = root.ref['id']
    models = {}
    for i, p in panels.items():
        if ref in p._models:
            model = p._models[ref][0]
        else:
            model = p._get_model(doc, root, parent, comm)
        model.margin = (0, 0, 0, 0)
        models[i] = model
    return models