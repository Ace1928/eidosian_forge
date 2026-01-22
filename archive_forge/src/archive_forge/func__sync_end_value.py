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
def _sync_end_value(self, event):
    if event.name == 'value':
        start = self.value[0] if self.value else self.start
    else:
        start = self.value_throttled[0] if self.value_throttled else self.start
    with param.edit_constant(self):
        self.param.update(**{event.name: (start, event.new)})