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
@param.depends('object', watch=True)
def _update_responsive(self):
    from holoviews import HoloMap, Store
    from holoviews.plotting import Plot
    obj = self.object
    if isinstance(obj, Plot):
        if 'responsive' in obj.param:
            wresponsive = obj.responsive and (not obj.width)
            hresponsive = obj.responsive and (not obj.height)
        elif 'sizing_mode' in obj.param:
            mode = obj.sizing_mode
            if mode:
                wresponsive = '_width' in mode or '_both' in mode
                hresponsive = '_height' in mode or '_both' in mode
            else:
                wresponsive = hresponsive = False
        else:
            wresponsive = hresponsive = False
        self._width_responsive = wresponsive
        self._height_responsive = hresponsive
        return
    obj = obj.last if isinstance(obj, HoloMap) else obj
    if obj is None or not Store.renderers:
        return
    backend = self.backend or Store.current_backend
    renderer = self.renderer or Store.renderers[backend]
    opts = obj.opts.get('plot', backend=backend).kwargs
    plot_cls = renderer.plotting_class(obj)
    if backend == 'matplotlib':
        self._width_responsive = self._height_responsive = False
    elif backend == 'plotly':
        responsive = opts.get('responsive', None)
        width = opts.get('width', None)
        height = opts.get('height', None)
        self._width_responsive = responsive and (not width)
        self._height_responsive = responsive and (not height)
    elif 'sizing_mode' in plot_cls.param:
        mode = opts.get('sizing_mode')
        if mode:
            self._width_responsive = '_width' in mode or '_both' in mode
            self._height_responsive = '_height' in mode or '_both' in mode
        else:
            self._width_responsive = False
            self._height_responsive = False
    else:
        responsive = opts.get('responsive', None)
        width = opts.get('width', None)
        frame_width = opts.get('frame_width', None)
        height = opts.get('height', None)
        frame_height = opts.get('frame_height', None)
        self._width_responsive = responsive and (not width) and (not frame_width)
        self._height_responsive = responsive and (not height) and (not frame_height)