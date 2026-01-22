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
class SyncableData(Reactive):
    """
    A baseclass for components which sync one or more data parameters
    with the frontend via a ColumnDataSource.
    """
    selection = param.List(default=[], item_type=int, doc='\n        The currently selected rows in the data.')
    _data_params: ClassVar[List[str]] = []
    _rename: ClassVar[Mapping[str, str | None]] = {'selection': None}
    __abstract = True

    def __init__(self, **params):
        super().__init__(**params)
        self._data = None
        self._processed = None
        callbacks = [self.param.watch(self._validate, self._data_params)]
        if self._data_params:
            callbacks.append(self.param.watch(self._update_cds, self._data_params))
        callbacks.append(self.param.watch(self._update_selected, 'selection'))
        self._internal_callbacks += callbacks
        self._validate()
        self._update_cds()

    def _validate(self, *events: param.parameterized.Event) -> None:
        """
        Allows implementing validation for the data parameters.
        """

    def _get_data(self) -> Tuple[TData, 'DataDict']:
        """
        Implemented by subclasses converting data parameter(s) into
        a ColumnDataSource compatible data dictionary.

        Returns
        -------
        processed: object
          Raw data after pre-processing (e.g. after filtering)
        data: dict
          Dictionary of columns used to instantiate and update the
          ColumnDataSource
        """

    def _update_column(self, column: str, array: np.ndarray | List) -> None:
        """
        Implemented by subclasses converting changes in columns to
        changes in the data parameter.

        Parameters
        ----------
        column: str
          The name of the column to update.
        array: numpy.ndarray
          The array data to update the column with.
        """
        data = getattr(self, self._data_params[0])
        data[column] = array
        if self._processed is not None:
            self._processed[column] = array

    def _update_data(self, data: TData) -> None:
        self.param.update(**{self._data_params[0]: data})

    def _manual_update(self, events: Tuple[param.parameterized.Event, ...], model: Model, doc: Document, root: Model, parent: Optional[Model], comm: Comm) -> None:
        for event in events:
            if event.type == 'triggered' and self._updating:
                continue
            elif hasattr(self, '_update_' + event.name):
                getattr(self, '_update_' + event.name)(model)

    @updating
    def _update_cds(self, *events: param.parameterized.Event) -> None:
        self._processed, self._data = self._get_data()
        msg = {'data': self._data}
        named_events = {event.name: event for event in events}
        for ref, (m, _) in self._models.items():
            self._apply_update(named_events, msg, m.source, ref)

    @updating
    def _update_selected(self, *events: param.parameterized.Event, indices: Optional[List[int]]=None) -> None:
        indices = self.selection if indices is None else indices
        msg = {'indices': indices}
        named_events = {event.name: event for event in events}
        for ref, (m, _) in self._models.items():
            self._apply_update(named_events, msg, m.source.selected, ref)

    def _apply_stream(self, ref: str, model: Model, stream: 'DataDict', rollover: Optional[int]) -> None:
        self._changing[ref] = ['data']
        try:
            model.source.stream(stream, rollover)
        finally:
            del self._changing[ref]

    @updating
    def _stream(self, stream: 'DataDict', rollover: Optional[int]=None) -> None:
        self._processed, _ = self._get_data()
        for ref, (m, _) in self._models.items():
            if ref not in state._views or ref in state._fake_roots:
                continue
            viewable, root, doc, comm = state._views[ref]
            if comm or not doc.session_context or state._unblocked(doc):
                with unlocked():
                    m.source.stream(stream, rollover)
                if comm and 'embedded' not in root.tags:
                    push(doc, comm)
            else:
                cb = partial(self._apply_stream, ref, m, stream, rollover)
                doc.add_next_tick_callback(cb)

    def _apply_patch(self, ref: str, model: Model, patch: 'Patches') -> None:
        self._changing[ref] = ['data']
        try:
            model.source.patch(patch)
        finally:
            del self._changing[ref]

    @updating
    def _patch(self, patch: 'Patches') -> None:
        for ref, (m, _) in self._models.items():
            if ref not in state._views or ref in state._fake_roots:
                continue
            viewable, root, doc, comm = state._views[ref]
            if comm or not doc.session_context or state._unblocked(doc):
                with unlocked():
                    m.source.patch(patch)
                if comm and 'embedded' not in root.tags:
                    push(doc, comm)
            else:
                cb = partial(self._apply_patch, ref, m, patch)
                doc.add_next_tick_callback(cb)

    def _update_manual(self, *events: param.parameterized.Event) -> None:
        """
        Skip events triggered internally
        """
        processed_events = []
        for e in events:
            if e.name == self._data_params[0] and e.type == 'triggered' and self._updating:
                continue
            processed_events.append(e)
        super()._update_manual(*processed_events)

    def stream(self, stream_value: 'pd.DataFrame' | 'pd.Series' | Dict, rollover: Optional[int]=None, reset_index: bool=True) -> None:
        """
        Streams (appends) the `stream_value` provided to the existing
        value in an efficient manner.

        Arguments
        ---------
        stream_value: (pd.DataFrame | pd.Series | Dict)
          The new value(s) to append to the existing value.
        rollover: (int | None, default=None)
           A maximum column size, above which data from the start of
           the column begins to be discarded. If None, then columns
           will continue to grow unbounded.
        reset_index (bool, default=True):
          If True and the stream_value is a DataFrame, then its index
          is reset. Helps to keep the index unique and named `index`.

        Raises
        ------
        ValueError: Raised if the stream_value is not a supported type.

        Examples
        --------

        Stream a Series to a DataFrame
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> stream_value = pd.Series({"x": 4, "y": "d"})
        >>> obj.stream(stream_value)
        >>> obj.value.to_dict("list")
        {'x': [1, 2, 4], 'y': ['a', 'b', 'd']}

        Stream a Dataframe to a Dataframe
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> stream_value = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
        >>> obj.stream(stream_value)
        >>> obj.value.to_dict("list")
        {'x': [1, 2, 3, 4], 'y': ['a', 'b', 'c', 'd']}

        Stream a Dictionary row to a DataFrame
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> tabulator = DataComponent(value)
        >>> stream_value = {"x": 4, "y": "d"}
        >>> obj.stream(stream_value)
        >>> obj.value.to_dict("list")
        {'x': [1, 2, 4], 'y': ['a', 'b', 'd']}

        Stream a Dictionary of Columns to a Dataframe
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> stream_value = {"x": [3, 4], "y": ["c", "d"]}
        >>> obj.stream(stream_value)
        >>> obj.value.to_dict("list")
        {'x': [1, 2, 3, 4], 'y': ['a', 'b', 'c', 'd']}
        """
        if 'pandas' in sys.modules:
            import pandas as pd
        else:
            pd = None
        if pd and isinstance(stream_value, pd.DataFrame):
            if isinstance(self._processed, dict):
                self.stream(stream_value.to_dict(), rollover)
                return
            if reset_index:
                value_index_start = self._processed.index.max() + 1
                stream_value = stream_value.reset_index(drop=True)
                stream_value.index += value_index_start
            combined = pd.concat([self._processed, stream_value])
            if rollover is not None:
                combined = combined.iloc[-rollover:]
            with param.discard_events(self):
                self._update_data(combined)
            try:
                self._updating = True
                self.param.trigger(self._data_params[0])
            finally:
                self._updating = False
            self._stream(stream_value, rollover)
        elif pd and isinstance(stream_value, pd.Series):
            if isinstance(self._processed, dict):
                self.stream({k: [v] for k, v in stream_value.to_dict().items()}, rollover)
                return
            value_index_start = self._processed.index.max() + 1
            self._processed.loc[value_index_start] = stream_value
            with param.discard_events(self):
                self._update_data(self._processed)
            self._stream(self._processed.iloc[-1:], rollover)
        elif isinstance(stream_value, dict):
            if isinstance(self._processed, dict):
                if not all((col in stream_value for col in self._data)):
                    raise ValueError('Stream update must append to all columns.')
                for col, array in stream_value.items():
                    combined = np.concatenate([self._data[col], array])
                    if rollover is not None:
                        combined = combined[-rollover:]
                    self._update_column(col, combined)
                self._stream(stream_value, rollover)
            else:
                try:
                    stream_value = pd.DataFrame(stream_value)
                except ValueError:
                    stream_value = pd.Series(stream_value)
                self.stream(stream_value)
        else:
            raise ValueError('The stream value provided is not a DataFrame, Series or Dict!')

    def patch(self, patch_value: 'pd.DataFrame' | 'pd.Series' | Dict) -> None:
        """
        Efficiently patches (updates) the existing value with the `patch_value`.

        Arguments
        ---------
        patch_value: (pd.DataFrame | pd.Series | Dict)
          The value(s) to patch the existing value with.

        Raises
        ------
        ValueError: Raised if the patch_value is not a supported type.

        Examples
        --------

        Patch a DataFrame with a Dictionary row.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> patch_value = {"x": [(0, 3)]}
        >>> obj.patch(patch_value)
        >>> obj.value.to_dict("list")
        {'x': [3, 2], 'y': ['a', 'b']}

        Patch a Dataframe with a Dictionary of Columns.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> patch_value = {"x": [(slice(2), (3,4))], "y": [(1,'d')]}
        >>> obj.patch(patch_value)
        >>> obj.value.to_dict("list")
        {'x': [3, 4], 'y': ['a', 'd']}

        Patch a DataFrame with a Series. Please note the index is used in the update.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> patch_value = pd.Series({"index": 1, "x": 4, "y": "d"})
        >>> obj.patch(patch_value)
        >>> obj.value.to_dict("list")
        {'x': [1, 4], 'y': ['a', 'd']}

        Patch a Dataframe with a Dataframe. Please note the index is used in the update.
        >>> value = pd.DataFrame({"x": [1, 2], "y": ["a", "b"]})
        >>> obj = DataComponent(value)
        >>> patch_value = pd.DataFrame({"x": [3, 4], "y": ["c", "d"]})
        >>> obj.patch(patch_value)
        >>> obj.value.to_dict("list")
        {'x': [3, 4], 'y': ['c', 'd']}
        """
        if self._processed is None or isinstance(patch_value, dict):
            self._patch(patch_value)
            return
        if 'pandas' in sys.modules:
            import pandas as pd
        else:
            pd = None
        data = getattr(self, self._data_params[0])
        patch_value_dict: Patches = {}
        if pd and isinstance(patch_value, pd.DataFrame):
            for column in patch_value.columns:
                patch_value_dict[column] = []
                for index in patch_value.index:
                    patch_value_dict[column].append((index, patch_value.loc[index, column]))
            self.patch(patch_value_dict)
        elif pd and isinstance(patch_value, pd.Series):
            if 'index' in patch_value:
                patch_value_dict = {k: [(patch_value['index'], v)] for k, v in patch_value.items()}
                patch_value_dict.pop('index')
            else:
                patch_value_dict = {patch_value.name: [(index, value) for index, value in patch_value.items()]}
            self.patch(patch_value_dict)
        elif isinstance(patch_value, dict):
            for k, v in patch_value.items():
                for index, patch in v:
                    if pd and isinstance(self._processed, pd.DataFrame):
                        data.loc[index, k] = patch
                    else:
                        data[k][index] = patch
            self._updating = True
            try:
                self._patch(patch_value)
            finally:
                self._updating = False
        else:
            raise ValueError(f'Patching with a patch_value of type {type(patch_value).__name__} is not supported. Please provide a DataFrame, Series or Dict.')