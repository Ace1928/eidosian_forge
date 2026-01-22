from __future__ import annotations
import ast
import json
from base64 import b64decode
from datetime import date, datetime
from typing import (
import numpy as np
import param
from bokeh.models.formatters import TickFormatter
from bokeh.models.widgets import (
from ..config import config
from ..layout import Column, Panel
from ..models import (
from ..util import param_reprs, try_datetime64_to_datetime
from .base import CompositeWidget, Widget
class FloatInput(_SpinnerBase, _FloatInputBase):
    """
    The `FloatInput` allows selecting a floating point value using a spinbox.

    It behaves like a slider except that the lower and upper bounds are
    optional and a specific value can be entered. The value can be changed
    using the keyboard (up, down, page up, page down), mouse wheel and arrow
    buttons.

    Reference: https://panel.holoviz.org/reference/widgets/FloatInput.html

    :Example:

    >>> FloatInput(name='Value', value=5., step=1e-1, start=0, end=10)
    """
    placeholder = param.String(default='', doc='\n        Placeholder when the value is empty.')
    step = param.Number(default=0.1, doc='\n        The step size.')
    value_throttled = param.Number(default=None, constant=True, doc='\n        The current value. Updates only on `<enter>` or when the widget looses focus.')
    _rename: ClassVar[Mapping[str, str | None]] = {'start': 'low', 'end': 'high'}

    def _process_param_change(self, msg):
        if msg.get('value', False) is None:
            msg['value'] = float('NaN')
        if msg.get('value_throttled', False) is None:
            msg['value_throttled'] = float('NaN')
        return super()._process_param_change(msg)

    def _process_property_change(self, msg):
        if msg.get('value', False) and np.isnan(msg['value']):
            msg['value'] = None
        if msg.get('value_throttled', False) and np.isnan(msg['value_throttled']):
            msg['value_throttled'] = None
        return super()._process_property_change(msg)