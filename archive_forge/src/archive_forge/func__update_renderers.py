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
def _update_renderers(self, model):
    model.renderers = []
    properties = self._get_properties(model.document)
    data, needle_data = self._get_data(properties)
    bar_source = ColumnDataSource(data=data, name='bar_source')
    needle_source = ColumnDataSource(data=needle_data, name='needle_source')
    if self.horizontal:
        model.hbar(y=0.1, left='y0', right='y1', height=1, color='color', source=bar_source)
        wedge_params = {'y': 0.5, 'x': 'y', 'angle': np.deg2rad(180)}
        text_params = {'y': -0.4, 'x': 0, 'text_align': 'left', 'text_baseline': 'top'}
    else:
        model.vbar(x=0.1, bottom='y0', top='y1', width=0.9, color='color', source=bar_source)
        wedge_params = {'x': 0.5, 'y': 'y', 'angle': np.deg2rad(90)}
        text_params = {'x': -0.4, 'y': 0, 'text_align': 'left', 'text_baseline': 'bottom', 'angle': np.deg2rad(90)}
    model.scatter(fill_color=self.needle_color, line_color=self.needle_color, source=needle_source, name='needle_renderer', marker='triangle', size=int(self.width / 8), level='overlay', **wedge_params)
    value_size = self.value_size or f'{self.width / 8}px'
    model.text(text='text', source=needle_source, text_font_size=value_size, **text_params)