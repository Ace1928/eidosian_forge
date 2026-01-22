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
def _get_style_data(self, recompute=True):
    if self.value is None or self.style is None or self.value.empty:
        return {}
    df = self._processed
    if recompute:
        try:
            self._computed_styler = styler = df.style
        except Exception:
            self._computed_styler = None
            return {}
        if styler is None:
            return {}
        styler._todo = styler_update(self.style, df)
        try:
            styler._compute()
        except Exception:
            styler._todo = []
    else:
        styler = self._computed_styler
    if styler is None:
        return {}
    offset = 1 + len(self.indexes) + int(self.selectable in ('checkbox', 'checkbox-single')) + int(bool(self.row_content))
    if self.pagination == 'remote':
        start = (self.page - 1) * self.page_size
        end = start + self.page_size
    column_mapper = {}
    frozen_cols = self.frozen_columns
    column_mapper = {}
    if isinstance(frozen_cols, list):
        nfrozen = len(frozen_cols)
        non_frozen = [col for col in df.columns if col not in frozen_cols]
        for i, col in enumerate(df.columns):
            if col in frozen_cols:
                column_mapper[i] = frozen_cols.index(col) - len(self.indexes)
            else:
                column_mapper[i] = nfrozen + non_frozen.index(col)
    elif isinstance(frozen_cols, dict):
        left_cols = [col for col, p in frozen_cols.items() if p in 'left']
        right_cols = [col for col, p in frozen_cols.items() if p in 'right']
        non_frozen = [col for col in df.columns if col not in frozen_cols]
        for i, col in enumerate(df.columns):
            if col in left_cols:
                column_mapper[i] = left_cols.index(col) - len(self.indexes)
            elif col in right_cols:
                column_mapper[i] = len(left_cols) + len(non_frozen) + right_cols.index(col)
            else:
                column_mapper[i] = len(left_cols) + non_frozen.index(col)
    styles = {}
    for (r, c), s in styler.ctx.items():
        if self.pagination == 'remote':
            if r < start or r >= end:
                continue
            else:
                r -= start
        if r not in styles:
            styles[int(r)] = {}
        c = column_mapper.get(int(c), int(c))
        styles[int(r)][offset + c] = s
    return {'id': uuid.uuid4().hex, 'data': styles}