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
def _update_style(self, *events):
    style = {p: getattr(self, p) for p in self._style_params}
    margin = style.pop('margin')
    if isinstance(margin, tuple):
        if len(margin) == 2:
            t = b = margin[0]
            r = l = margin[1]
        else:
            t, r, b, l = margin
    else:
        t = r = b = l = margin
    text_margin = (t, 0, 0, l)
    slider_margin = (0, r, b, l)
    text_style = {k: v for k, v in style.items() if k not in ('style', 'orientation')}
    text_style['visible'] = self.show_value and text_style['visible']
    self._text.param.update(margin=text_margin, **text_style)
    self._slider.param.update(margin=slider_margin, **style)
    if self.width:
        style['width'] = self.width + l + r
    col_style = {k: v for k, v in style.items() if k != 'orientation'}
    self._composite.param.update(**col_style)