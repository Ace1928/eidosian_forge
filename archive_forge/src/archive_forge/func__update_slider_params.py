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
def _update_slider_params(self, *events):
    style = {e.name: e.new for e in events}
    disabled = style.get('disabled', None)
    if disabled is False:
        if len(self.values) in (0, 1):
            self.param.warning('A DiscreteSlider can only be disabled if it has more than 1 option.')
            end = 1
            del style['disabled']
        else:
            end = len(self.options) - 1
        style['end'] = end
    self._slider.param.update(**style)