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
class _IntInputBase(_NumericInputBase):
    value = param.Integer(default=0, allow_None=True, doc='\n        The current value of the spinner.')
    start = param.Integer(default=None, allow_None=True, doc='\n        Optional minimum allowable value.')
    end = param.Integer(default=None, allow_None=True, doc='\n        Optional maximum allowable value.')
    mode = param.String(default='int', constant=True, doc='\n        Define the type of number which can be enter in the input')
    __abstract = True