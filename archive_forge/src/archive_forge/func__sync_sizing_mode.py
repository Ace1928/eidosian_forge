from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
def _sync_sizing_mode(self, plot):
    state = plot.state
    backend = plot.renderer.backend
    if backend == 'bokeh':
        params = {'sizing_mode': state.sizing_mode, 'width': state.width, 'height': state.height}
    elif backend == 'matplotlib':
        params = {'sizing_mode': None, 'width': None, 'height': None}
    elif backend == 'plotly':
        if state.get('config', {}).get('responsive'):
            sizing_mode = 'stretch_both'
        else:
            sizing_mode = None
        params = {'sizing_mode': sizing_mode, 'width': None, 'height': None}
    self._syncing_props = True
    try:
        self.param.update({k: v for k, v in params.items() if k not in self._overrides})
        if backend != 'bokeh':
            return
        plot_props = plot.state.properties()
        props = {o: getattr(self, o) for o in self._overrides if o in plot_props}
        if props:
            plot.state.update(**props)
    finally:
        self._syncing_props = False