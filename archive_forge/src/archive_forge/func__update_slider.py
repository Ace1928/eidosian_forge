from __future__ import annotations
import datetime as dt
from typing import (
import numpy as np
import param
from bokeh.models import CustomJS
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from bokeh.models.widgets.sliders import NumericalSlider as _BkNumericalSlider
from param.parameterized import resolve_value
from ..config import config
from ..io import state
from ..io.resources import CDN_DIST
from ..layout import Column, Panel, Row
from ..util import (
from ..viewable import Layoutable
from ..widgets import FloatInput, IntInput
from .base import CompositeWidget, Widget
from .input import StaticText
@param.depends('start', 'end', 'step', 'bar_color', 'direction', 'show_value', 'tooltips', 'name', 'format', watch=True)
def _update_slider(self):
    self._slider.param.update(format=self.format, start=self.start, end=self.end, step=self.step, bar_color=self.bar_color, direction=self.direction, show_value=self.show_value, tooltips=self.tooltips)
    self._start_edit.step = self.step
    self._end_edit.step = self.step