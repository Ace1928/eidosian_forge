from __future__ import annotations
import datetime as dt
import difflib
import inspect
import logging
import re
import sys
import textwrap
from collections import Counter, defaultdict, namedtuple
from functools import lru_cache, partial
from pprint import pformat
from typing import (
import numpy as np
import param
from bokeh.core.property.descriptors import UnsetValueError
from bokeh.model import DataModel
from bokeh.models import ImportedStyleSheet
from packaging.version import Version
from param.parameterized import (
from .io.document import unlocked
from .io.model import hold
from .io.notebook import push
from .io.resources import (
from .io.state import set_curdoc, state
from .models.reactive_html import (
from .util import (
from .viewable import Layoutable, Renderable, Viewable
class ReactiveData(SyncableData):
    """
    An extension of SyncableData which bi-directionally syncs a data
    parameter between frontend and backend using a ColumnDataSource.
    """
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)

    def _update_selection(self, indices: List[int]) -> None:
        self.selection = indices

    def _convert_column(self, values: np.ndarray, old_values: np.ndarray | 'pd.Series') -> np.ndarray | List:
        dtype = old_values.dtype
        converted: List | np.ndarray | None = None
        if dtype.kind == 'M':
            if values.dtype.kind in 'if':
                NATs = np.isnan(values)
                converted = np.where(NATs, np.nan, values * 1000000.0).astype(dtype)
        elif dtype.kind == 'O':
            if all((isinstance(ov, dt.date) for ov in old_values)) and (not all((isinstance(iv, dt.date) for iv in values))):
                new_values = []
                for iv in values:
                    if isinstance(iv, dt.datetime):
                        iv = iv.date()
                    elif not isinstance(iv, dt.date):
                        iv = dt.date.fromtimestamp(iv / 1000)
                    new_values.append(iv)
                converted = new_values
        elif 'pandas' in sys.modules:
            import pandas as pd
            if Version(pd.__version__) >= Version('1.1.0'):
                from pandas.core.arrays.masked import BaseMaskedDtype
                if isinstance(dtype, BaseMaskedDtype):
                    values = [dtype.na_value if v == '<NA>' else v for v in values]
            converted = pd.Series(values).astype(dtype).values
        else:
            converted = values.astype(dtype)
        return values if converted is None else converted

    def _process_data(self, data: Mapping[str, List | Dict[int, Any] | np.ndarray]) -> None:
        if self._updating:
            return
        old_raw, old_data = self._get_data()
        old_raw = old_raw.copy()
        if hasattr(old_raw, 'columns'):
            columns = list(old_raw.columns)
        else:
            columns = list(old_raw)
        updated = False
        for col, values in data.items():
            col = self._renamed_cols.get(col, col)
            if col in self.indexes or col not in columns:
                continue
            if isinstance(values, dict):
                sorted_values = sorted(values.items(), key=lambda it: int(it[0]))
                values = [v for _, v in sorted_values]
            values = self._convert_column(np.asarray(values), old_raw[col])
            isequal = None
            if hasattr(old_raw, 'columns') and isinstance(values, np.ndarray):
                try:
                    isequal = np.array_equal(old_raw[col], values, equal_nan=True)
                except Exception:
                    pass
            if isequal is None:
                try:
                    isequal = (old_raw[col] == values).all()
                except Exception:
                    isequal = False
            if not isequal:
                self._update_column(col, values)
                updated = True
        if not updated:
            return
        self._updating = True
        old_data = getattr(self, self._data_params[0])
        try:
            if old_data is self.value:
                with param.discard_events(self):
                    self.value = old_raw
                self.value = old_data
            else:
                self.param.trigger('value')
        finally:
            self._updating = False
        if old_data is not self.value:
            self._update_cds()

    def _process_events(self, events: Dict[str, Any]) -> None:
        if 'data' in events:
            self._process_data(events.pop('data'))
        if 'indices' in events:
            self._updating = True
            try:
                self._update_selection(events.pop('indices'))
            finally:
                self._updating = False
        super(ReactiveData, self)._process_events(events)