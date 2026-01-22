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
class BooleanStatus(BooleanIndicator):
    """
    The `BooleanStatus` is a boolean indicator providing a visual
    representation of a boolean status as filled or non-filled circle.

    If the value is set to `True` the indicator will be filled while
    setting it to `False` will cause it to be non-filled.

    Reference: https://panel.holoviz.org/reference/indicators/BooleanStatus.html

    :Example:

    >>> BooleanStatus(value=True, color='primary', width=100, height=100)
    """
    color = param.ObjectSelector(default='dark', objects=['primary', 'secondary', 'success', 'info', 'danger', 'warning', 'light', 'dark'], doc="\n        The color of the circle, one of 'primary', 'secondary', 'success', 'info', 'danger',\n        'warning', 'light', 'dark'")
    height = param.Integer(default=20, doc='\n        height of the circle.')
    width = param.Integer(default=20, doc='\n        Width of the circle.')
    value = param.Boolean(default=False, doc='\n        Whether the indicator is active or not.')
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'value': None, 'color': None}
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/booleanstatus.css']
    _widget_type: ClassVar[Type[Model]] = HTML

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        value = msg.pop('value', None)
        color = msg.pop('color', None)
        if value is None and (not color):
            return msg
        msg['css_classes'] = ['dot-filled', self.color] if self.value else ['dot']
        return msg