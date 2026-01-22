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