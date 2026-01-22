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
class TooltipIcon(Widget):
    """
    The `TooltipIcon` displays a small `?` icon. When you hover over the `?` icon, the `value`
    will display.

    Use the `TooltipIcon` to provide

    - helpful information to users without taking up a lot of screen space
    - tooltips next to Panel widgets that do not support tooltips yet.

    Reference: https://panel.holoviz.org/reference/indicators/TooltipIcon.html

    :Example:

    >>> pn.widgets.TooltipIcon(value="This is a simple tooltip by using a string")
    """
    align = Align(default='center', doc='\n        Whether the object should be aligned with the start, end or\n        center of its container. If set as a tuple it will declare\n        (vertical, horizontal) alignment.')
    value = param.ClassSelector(default='Description', class_=(str, Tooltip), doc='\n        The description in the tooltip.')
    _rename: ClassVar[Mapping[str, str | None]] = {'name': None, 'value': 'description'}
    _widget_type = _BkTooltipIcon