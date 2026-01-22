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
def _convert_to_datetime(self, v):
    if v is None:
        return
    if isinstance(v, Iterable) and (not isinstance(v, str)):
        container_type = type(v)
        return container_type((self._convert_to_datetime(vv) for vv in v))
    v = try_datetime64_to_datetime(v)
    if isinstance(v, datetime):
        return v
    elif isinstance(v, date):
        return datetime(v.year, v.month, v.day)
    elif isinstance(v, str):
        return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
    else:
        raise ValueError(f'Could not convert {v} to datetime')