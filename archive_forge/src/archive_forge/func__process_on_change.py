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
def _process_on_change(self, event: param.parameterized.Event):
    old, new = (event.old, event.new)
    for model in (old if isinstance(old, dict) else {}).values():
        if not isinstance(model, (CellEditor, CellFormatter)):
            continue
        change_fn = self._editor_change if isinstance(model, CellEditor) else self._formatter_change
        for prop in model.properties() - Model.properties():
            try:
                model.remove_on_change(prop, change_fn)
            except ValueError:
                pass
    for model in (new if isinstance(new, dict) else {}).values():
        if not isinstance(model, (CellEditor, CellFormatter)):
            continue
        change_fn = self._editor_change if isinstance(model, CellEditor) else self._formatter_change
        for prop in model.properties() - Model.properties():
            model.on_change(prop, change_fn)