from __future__ import annotations
import asyncio
import math
import os
import sys
import time
from math import pi
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, FixedTicker, Tooltip
from bokeh.plotting import figure
from tqdm.asyncio import tqdm as _tqdm
from .._param import Align
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..models import (
from ..pane.markup import Str
from ..reactive import SyncableData
from ..util import PARAM_NAME_PATTERN, escape, updating
from ..viewable import Viewable
from .base import Widget
class Tqdm(Indicator):
    """
    The `Tqdm` indicator wraps the well known `tqdm` progress
    indicator and displays the progress towards some target in your
    Panel app.

    Reference: https://panel.holoviz.org/reference/indicators/Tqdm.html

    :Example:

    >>> tqdm = Tqdm()
    >>> for i in tqdm(range(0,10), desc="My loop", leave=True, colour='#666666'):
    ...     time.sleep(timeout)
    """
    value = param.Integer(default=0, bounds=(-1, None), doc='\n        The current value of the progress bar. If set to -1 the progress\n        bar will be indeterminate and animate depending on the active\n        parameter.')
    layout = param.ClassSelector(class_=(Column, Row), precedence=-1, constant=True, doc='\n        The layout for the text and progress indicator.')
    lock = param.ClassSelector(class_=object, default=None, doc='The `multithreading.Lock` or `multiprocessing.Lock` object to be used by Tqdm.')
    max = param.Integer(default=100, doc='The maximum value of the progress bar.')
    progress = param.ClassSelector(class_=Progress, allow_refs=False, precedence=-1, doc='\n        The Progress indicator used to display the progress.')
    text = param.String(default='', doc='\n        The current tqdm style progress text.')
    text_pane = param.ClassSelector(class_=Str, precedence=-1, doc='\n        The pane to display the text to.')
    margin = param.Parameter(default=0, doc='\n        Allows to create additional space around the component. May\n        be specified as a two-tuple of the form (vertical, horizontal)\n        or a four-tuple (top, right, bottom, left).')
    width = param.Integer(default=400, bounds=(0, None), doc='\n        The width of the component (in pixels). This can be either\n        fixed or preferred width, depending on width sizing policy.')
    write_to_console = param.Boolean(default=False, doc='\n        Whether or not to also write to the console.')
    _layouts: ClassVar[Dict[Type[Panel], str]] = {Row: 'row', Column: 'column'}
    _rename: ClassVar[Mapping[str, str | None]] = {'value': None, 'min': None, 'max': None, 'text': None, 'name': 'name'}

    def __init__(self, **params):
        layout = params.pop('layout', 'column')
        layout = self._layouts.get(layout, layout)
        if 'text_pane' not in params:
            sizing_mode = 'stretch_width' if layout == 'column' else 'fixed'
            params['text_pane'] = Str(None, min_height=20, min_width=280, sizing_mode=sizing_mode, margin=MARGIN['text_pane'][layout])
        if 'progress' not in params:
            params['progress'] = Progress(active=False, sizing_mode='stretch_width', min_width=100, margin=MARGIN['progress'][layout])
        layout_params = {p: params.get(p, getattr(self, p)) for p in Viewable.param}
        if layout == 'row' or layout is Row:
            params['layout'] = Row(params['progress'], params['text_pane'], **layout_params)
        else:
            params['layout'] = Column(params['text_pane'], params['progress'], **layout_params)
        super().__init__(**params)
        self.param.watch(self._update_layout, list(Viewable.param))
        self.progress.max = self.max
        self.progress.value = self.value
        self.text_pane.object = self.text
        try:
            from multiprocessing import Lock
            self._lock = params.pop('lock', Lock())
        except ImportError:
            self._lock = params.pop('lock', None)

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        model = self.layout._get_model(doc, root, parent, comm)
        root = root or model
        self._models[root.ref['id']] = (model, parent)
        return model

    def _cleanup(self, root: Model | None=None) -> None:
        super()._cleanup(root)
        self.layout._cleanup(root)

    def _update_layout(self, *events):
        self.layout.param.update(**{event.name: event.new for event in events})

    def get_lock(self) -> LockType:
        return self._lock

    def set_lock(self, lock: LockType) -> None:
        self._lock = lock

    @param.depends('text', watch=True)
    def _update_text(self):
        if self.text_pane:
            self.text_pane.object = self.text

    @param.depends('value', watch=True)
    def _update_value(self):
        if self.progress:
            self.progress.value = self.value

    @param.depends('max', watch=True)
    def _update_max(self):
        if self.progress:
            self.progress.max = self.max

    def __call__(self, *args, **kwargs):
        kwargs['indicator'] = self
        if not self.write_to_console:
            f = open(os.devnull, 'w')
            kwargs['file'] = f
        return ptqdm(*args, **kwargs)
    __call__.__doc__ = ptqdm.__doc__

    def pandas(self, *args, **kwargs):
        kwargs['indicator'] = self
        if not self.write_to_console and 'file' not in kwargs:
            f = open(os.devnull, 'w')
            kwargs['file'] = f
        return ptqdm.pandas(*args, **kwargs)

    def reset(self):
        """Resets the parameters"""
        self.value = self.param.value.default
        self.text = self.param.text.default