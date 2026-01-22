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
def _get_widgets(self):
    """Return name,widget boxes for all parameters (i.e., a property sheet)"""
    if self.expand_layout is Tabs:
        widgets = []
    elif self.show_name:
        widgets = [('_title', StaticText(value='<b>{0}</b>'.format(self.name)))]
    else:
        widgets = []
    widgets += [(pname, self.widget(pname)) for pname in self._ordered_params]
    return dict(widgets)