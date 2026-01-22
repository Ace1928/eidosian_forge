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
def _link_subobjects(self):
    for pname, widget in self._widgets.items():
        widgets = [widget] if isinstance(widget, Widget) else widget
        if not any((is_parameterized(getattr(w, 'value', None)) or any((is_parameterized(o) for o in getattr(w, 'options', []))) for w in widgets)):
            continue
        if isinstance(widgets, Row) and isinstance(widgets[1], Toggle):
            selector, toggle = (widgets[0], widgets[1])
        else:
            selector, toggle = (widget, None)

        def toggle_pane(change, parameter=pname):
            """Adds or removes subpanel from layout"""
            parameterized = getattr(self.object, parameter)
            existing = [p for p in self._expand_layout.objects if isinstance(p, Param) and p.object in recursive_parameterized(parameterized)]
            if not change.new:
                self._expand_layout[:] = [e for e in self._expand_layout.objects if e not in existing]
            elif change.new:
                kwargs = {k: v for k, v in self.param.values().items() if k not in ['name', 'object', 'parameters']}
                pane = Param(parameterized, name=parameterized.name, **kwargs)
                if isinstance(self._expand_layout, Tabs):
                    title = self.object.param[parameter].label
                    pane = (title, pane)
                self._expand_layout.append(pane)

        def update_pane(change, parameter=pname, toggle=toggle):
            """Adds or removes subpanel from layout"""
            layout = self._expand_layout
            existing = [p for p in layout.objects if isinstance(p, Param) and p.object is change.old]
            if toggle:
                toggle.disabled = not is_parameterized(change.new)
            if not existing:
                return
            elif is_parameterized(change.new):
                parameterized = change.new
                kwargs = {k: v for k, v in self.param.values().items() if k not in ['name', 'object', 'parameters']}
                pane = Param(parameterized, name=parameterized.name, **kwargs)
                layout[layout.objects.index(existing[0])] = pane
            else:
                layout.remove(existing[0])
        watchers = [selector.param.watch(update_pane, 'value')]
        if toggle:
            watchers.append(toggle.param.watch(toggle_pane, 'value'))
        self._internal_callbacks += watchers
        if self.expand:
            if self.expand_button:
                toggle.value = True
            else:
                toggle_pane(namedtuple('Change', 'new')(True))