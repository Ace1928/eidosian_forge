from __future__ import annotations
import datetime as dt
import sys
from enum import Enum
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from pyviz_comms import JupyterComm
from ..io.state import state
from ..reactive import ReactiveData
from ..util import datetime_types, lazy_load
from ..viewable import Viewable
from .base import ModelPane
class Perspective(ModelPane, ReactiveData):
    """
    The `Perspective` pane provides an interactive visualization component for
    large, real-time datasets built on the Perspective project.

    Reference: https://panel.holoviz.org/reference/panes/Perspective.html

    :Example:

    >>> Perspective(df, plugin='hypergrid', theme='pro-dark')
    """
    aggregates = param.Dict(default=None, nested_refs=True, doc='\n      How to aggregate. For example {"x": "distinct count"}')
    columns = param.List(default=None, nested_refs=True, doc='\n      A list of source columns to show as columns. For example ["x", "y"]')
    columns_config = param.Dict(default=None, nested_refs=True, doc='\n      Column configuration allowing specification of formatters, coloring\n      and a variety of other attributes for each column.')
    editable = param.Boolean(default=True, allow_None=True, doc='\n      Whether items are editable.')
    expressions = param.ClassSelector(class_=(dict, list), default=None, nested_refs=True, doc='\n      A list of expressions computing new columns from existing columns.\n      For example [""x"+"index""]')
    split_by = param.List(default=None, nested_refs=True, doc='\n      A list of source columns to pivot by. For example ["x", "y"]')
    filters = param.List(default=None, nested_refs=True, doc='\n      How to filter. For example [["x", "<", 3],["y", "contains", "abc"]]')
    min_width = param.Integer(default=420, bounds=(0, None), doc='\n        Minimal width of the component (in pixels) if width is adjustable.')
    object = param.Parameter(doc='\n      The plot data declared as a dictionary of arrays or a DataFrame.')
    group_by = param.List(default=None, doc='\n      A list of source columns to group by. For example ["x", "y"]')
    selectable = param.Boolean(default=True, allow_None=True, doc='\n      Whether items are selectable.')
    sort = param.List(default=None, doc='\n      How to sort. For example[["x","desc"]]')
    plugin = param.ObjectSelector(default=Plugin.GRID.value, objects=Plugin.options(), doc='\n      The name of a plugin to display the data. For example hypergrid or d3_xy_scatter.')
    plugin_config = param.Dict(default={}, nested_refs=True, doc='\n      Configuration for the PerspectiveViewerPlugin.')
    settings = param.Boolean(default=True, doc='\n      Whether to show the settings menu.')
    toggle_config = param.Boolean(default=True, doc='\n      Whether to show the config menu.')
    theme = param.ObjectSelector(default='pro', objects=THEMES, doc='\n      The style of the PerspectiveViewer. For example pro-dark')
    title = param.String(default=None, doc='\n      Title for the Perspective viewer.')
    priority: ClassVar[float | bool | None] = None
    _bokeh_model: ClassVar[Type[Model] | None] = None
    _data_params: ClassVar[List[str]] = ['object']
    _rename: ClassVar[Mapping[str, str | None]] = {'selection': None}
    _updates: ClassVar[bool] = True

    @classmethod
    def applies(cls, object):
        if isinstance(object, dict) and all((isinstance(v, (list, np.ndarray)) for v in object.values())):
            return 0 if object else None
        elif 'pandas' in sys.modules:
            import pandas as pd
            if isinstance(object, pd.DataFrame):
                return 0
        return False

    def __init__(self, object=None, **params):
        click_handler = params.pop('on_click', None)
        self._on_click_callbacks = []
        super().__init__(object, **params)
        if click_handler:
            self.on_click(click_handler)

    def _get_data(self):
        if self.object is None:
            return ({}, {})
        if isinstance(self.object, dict):
            ncols = len(self.object)
            df = data = self.object
        else:
            df, kwargs = deconstruct_pandas(self.object)
            ncols = len(df.columns)
            data = {col: df[col].values for col in df.columns}
            if kwargs:
                self.param.update(**{k: v for k, v in kwargs.items() if getattr(self, k) is None})
        cols = set((self._as_digit(c) for c in df))
        if len(cols) != ncols:
            raise ValueError('Integer columns must be unique when converted to strings.')
        return (df, {str(k): v for k, v in data.items()})

    def _filter_properties(self, properties):
        ignored = list(Viewable.param)
        return [p for p in properties if p not in ignored]

    def _get_properties(self, doc, source=None):
        props = super()._get_properties(doc)
        if 'theme' in props and 'material' in props['theme']:
            props['theme'] = props['theme'].replace('material', 'pro')
        del props['object']
        if props.get('toggle_config'):
            props['height'] = self.height or 300
        else:
            props['height'] = self.height or 150
        if source is None:
            source = ColumnDataSource(data=self._data)
        else:
            source.data = self._data
        props['source'] = source
        props['schema'] = schema = {}
        for col, array in source.data.items():
            if not isinstance(array, np.ndarray):
                array = np.asarray(array)
            kind = array.dtype.kind
            if kind == 'M':
                schema[col] = 'datetime'
            elif kind in 'ui':
                schema[col] = 'integer'
            elif kind == 'b':
                schema[col] = 'boolean'
            elif kind == 'f':
                schema[col] = 'float'
            elif kind in 'sU':
                schema[col] = 'string'
            elif len(array):
                value = array[0]
                if isinstance(value, datetime_types) and type(value) is not dt.date:
                    schema[col] = 'datetime'
                elif isinstance(value, dt.date):
                    schema[col] = 'date'
                elif isinstance(value, str):
                    schema[col] = 'string'
                elif isinstance(value, (float, np.floating)):
                    schema[col] = 'float'
                elif isinstance(value, (int, np.integer)):
                    schema[col] = 'integer'
                else:
                    schema[col] = 'string'
            else:
                schema[col] = 'string'
        return props

    def _get_theme(self, theme, resources=None):
        from ..models.perspective import THEME_URL
        theme_url = f'{THEME_URL}{theme}.css'
        if self._bokeh_model is not None:
            self._bokeh_model.__css_raw__ = self._bokeh_model.__css_raw__[:5] + [theme_url]
        return theme_url

    def _process_param_change(self, params):
        if 'stylesheets' in params or 'theme' in params:
            self._get_theme(params.get('theme', self.theme))
            css = getattr(self._bokeh_model, '__css__', [])
            params['stylesheets'] = [ImportedStyleSheet(url=ss) for ss in css] + params.get('stylesheets', self.stylesheets)
        if 'theme' in params and 'material' in params['theme']:
            params['theme'] = params['theme'].replace('material', 'pro')
        props = super()._process_param_change(params)
        for p in ('columns', 'group_by', 'split_by'):
            if props.get(p):
                props[p] = [None if col is None else str(col) for col in props[p]]
        if props.get('sort'):
            props['sort'] = [[str(col), *args] for col, *args in props['sort']]
        if props.get('filters'):
            props['filters'] = [[str(col), *args] for col, *args in props['filters']]
        if props.get('aggregates'):
            props['aggregates'] = {str(col): agg for col, agg in props['aggregates'].items()}
        if isinstance(props.get('expressions'), list):
            props['expressions'] = {f'expression_{i}': exp for i, exp in enumerate(props['expressions'])}
        return props

    def _as_digit(self, col):
        if self._processed is None or col in self._processed or col is None:
            return col
        elif col.isdigit() and int(col) in self._processed:
            return int(col)
        return col

    def _process_property_change(self, msg):
        msg = super()._process_property_change(msg)
        for prop in ('columns', 'group_by', 'split_by'):
            if prop not in msg:
                continue
            msg[prop] = [self._as_digit(col) for col in msg[prop]]
        if msg.get('sort'):
            msg['sort'] = [[self._as_digit(col), *args] for col, *args in msg['sort']]
        if msg.get('filters'):
            msg['filters'] = [[self._as_digit(col), *args] for col, *args in msg['filters']]
        if msg.get('aggregates'):
            msg['aggregates'] = {self._as_digit(col): agg for col, agg in msg['aggregates'].items()}
        return msg

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        self._bokeh_model = lazy_load('panel.models.perspective', 'Perspective', isinstance(comm, JupyterComm), root)
        model = super()._get_model(doc, root, parent, comm)
        self._register_events('perspective-click', model=model, doc=doc, comm=comm)
        return model

    def _update(self, ref: str, model: Model) -> None:
        model.update(**self._get_properties(model.document, source=model.source))

    def _process_event(self, event):
        if event.event_name == 'perspective-click':
            for cb in self._on_click_callbacks:
                state.execute(partial(cb, event), schedule=False)

    def on_click(self, callback: Callable[[PerspectiveClickEvent], None]):
        """
        Register a callback to be executed when any row is clicked.
        The callback is given a PerspectiveClickEvent declaring the
        config, column names, and row values of the row that was
        clicked.

        Arguments
        ---------
        callback: (callable)
            The callback to run on edit events.
        """
        self._on_click_callbacks.append(callback)