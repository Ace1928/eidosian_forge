from __future__ import annotations
import re
import sys
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import lazy_load
from .base import ModelPane
@property
def _throttle(self):
    default = self.param.debounce.default
    if isinstance(self.debounce, dict):
        throttle = {sel: self.debounce.get(sel, default) for sel in self._selections}
    else:
        throttle = {sel: self.debounce or default for sel in self._selections}
    return throttle