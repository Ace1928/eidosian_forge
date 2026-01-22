from __future__ import annotations
import asyncio
import inspect
import itertools
import json
import os
import sys
import textwrap
import types
from collections import defaultdict, namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import (
import param
from param.parameterized import (
from param.reactive import rx
from .config import config
from .io import state
from .layout import (
from .pane import DataFrame as DataFramePane
from .pane.base import PaneBase, ReplacementPane
from .reactive import Reactive
from .util import (
from .util.checks import is_dataframe, is_mpl_axes, is_series
from .viewable import Layoutable, Viewable
from .widgets import (
from .widgets.button import _ButtonBase
class ReactiveExpr(PaneBase):
    """
    ReactiveExpr generates a UI for param.rx objects by rendering the
    widgets and outputs.
    """
    center = param.Boolean(default=False, doc='\n        Whether to center the output.')
    object = param.Parameter(default=None, allow_refs=False, doc='\n        The object being wrapped, which will be converted to a\n        Bokeh model.')
    show_widgets = param.Boolean(default=True, doc='\n        Whether to display the widget inputs.')
    widget_layout = param.Selector(objects=[WidgetBox, Row, Column], constant=True, default=WidgetBox, doc='\n        The layout object to display the widgets in.')
    widget_location = param.Selector(default='left_top', objects=['left', 'right', 'top', 'bottom', 'top_left', 'top_right', 'bottom_left', 'bottom_right', 'left_top', 'right_top', 'right_bottom'], doc='\n        The location of the widgets relative to the output\n        of the reactive expression.')
    priority: ClassVar[float | bool | None] = 1
    _layouts = {'left': (Row, ('start', 'center'), True), 'right': (Row, ('end', 'center'), False), 'top': (Column, ('center', 'start'), True), 'bottom': (Column, ('center', 'end'), False), 'top_left': (Column, 'start', True), 'top_right': (Column, ('end', 'start'), True), 'bottom_left': (Column, ('start', 'end'), False), 'bottom_right': (Column, 'end', False), 'left_top': (Row, 'start', True), 'left_bottom': (Row, ('start', 'end'), True), 'right_top': (Row, ('end', 'start'), False), 'right_bottom': (Row, 'end', False)}
    _unpack: ClassVar[bool] = False

    def __init__(self, object=None, **params):
        super().__init__(object=object, **params)
        self._update_layout()

    @param.depends('center', 'object', 'widget_layout', 'widget_location', watch=True)
    def _update_layout(self, *events):
        if self.object is None:
            self.layout[:] = []
        else:
            self.layout[:] = [self._generate_layout()]

    @classmethod
    def applies(cls, object):
        return isinstance(object, param.rx)

    @classmethod
    def _find_widgets(cls, op):
        widgets = []
        op_args = list(op['args']) + list(op['kwargs'].values())
        op_args = flatten(op_args)
        for op_arg in op_args:
            if isinstance(op_arg, Widget) and op_arg not in widgets:
                widgets.append(op_arg)
                continue
            if 'ipywidgets' in sys.modules:
                from ipywidgets import Widget as IPyWidget
                if isinstance(op_arg, IPyWidget) and op_arg not in widgets:
                    widgets.append(op_arg)
                    continue
            if isinstance(op_arg, param.Parameter) and isinstance(op_arg.owner, Widget) and (op_arg.owner not in widgets):
                widgets.append(op_arg.owner)
                continue
            if hasattr(op_arg, '_dinfo'):
                dinfo = op_arg._dinfo
                args = list(dinfo.get('dependencies', []))
                kwargs = dinfo.get('kw', {})
                nested_op = {'args': args, 'kwargs': kwargs}
            elif isinstance(op_arg, slice):
                nested_op = {'args': [op_arg.start, op_arg.stop, op_arg.step], 'kwargs': {}}
            elif isinstance(op_arg, (list, tuple)):
                nested_op = {'args': op_arg, 'kwargs': {}}
            elif isinstance(op_arg, dict):
                nested_op = {'args': (), 'kwargs': op_arg}
            elif isinstance(op_arg, param.rx):
                nested_op = {'args': op_arg._params, 'kwargs': {}}
            else:
                continue
            for widget in cls._find_widgets(nested_op):
                if widget not in widgets:
                    widgets.append(widget)
        return widgets

    @property
    def widgets(self):
        widgets = []
        if self.object is None:
            return []
        for p in self.object._fn_params:
            if isinstance(p.owner, Widget) and p.owner not in widgets:
                widgets.append(p.owner)
        operations = []
        prev = self.object
        while prev is not None:
            if prev._operation:
                operations.append(prev._operation)
            prev = prev._prev
        for op in operations[::-1]:
            for w in self._find_widgets(op):
                if w not in widgets:
                    widgets.append(w)
        return self.widget_layout(*widgets)

    def _get_model(self, doc: Document, root: Optional['Model']=None, parent: Optional['Model']=None, comm: Optional[Comm]=None) -> 'Model':
        return self.layout._get_model(doc, root, parent, comm)

    def _generate_layout(self):
        panel = ParamFunction(self.object._callback)
        if not self.show_widgets:
            return panel
        widget_box = self.widgets
        loc = self.widget_location
        layout, align, widget_first = self._layouts[loc]
        widget_box.align = align
        if not len(widget_box):
            if self.center:
                components = [HSpacer(), panel, HSpacer()]
            else:
                components = [panel]
            return Row(*components)
        items = (widget_box, panel) if widget_first else (panel, widget_box)
        if not self.center:
            if layout is Row:
                components = list(items)
            else:
                components = [layout(*items, sizing_mode=self.sizing_mode)]
        elif layout is Column:
            components = [HSpacer(), layout(*items, sizing_mode=self.sizing_mode), HSpacer()]
        elif loc.startswith('left'):
            components = [widget_box, HSpacer(), panel, HSpacer()]
        else:
            components = [HSpacer(), panel, HSpacer(), widget_box]
        return Row(*components)