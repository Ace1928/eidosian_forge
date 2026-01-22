from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
def _update_from_figure(self, event, *args, **kwargs):
    self._event = event
    try:
        self.param.trigger('object')
    finally:
        self._event = None