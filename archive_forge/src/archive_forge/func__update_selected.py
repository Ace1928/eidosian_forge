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
def _update_selected(self, *events: param.parameterized.Event, indices=None):
    kwargs = {}
    if self.pagination == 'remote' and self.value is not None:
        index = self.value.iloc[self.selection].index
        indices = []
        for ind in index.values:
            try:
                iloc = self._processed.index.get_loc(ind)
                self._validate_iloc(ind, iloc)
                indices.append((ind, iloc))
            except KeyError:
                continue
        nrows = self.page_size
        start = (self.page - 1) * nrows
        end = start + nrows
        p_range = self._processed.index[start:end]
        kwargs['indices'] = [iloc - start for ind, iloc in indices if ind in p_range]
    super()._update_selected(*events, **kwargs)