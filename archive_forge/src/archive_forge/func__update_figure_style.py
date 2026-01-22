from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
@param.depends('restyle_data', watch=True)
def _update_figure_style(self):
    if self._figure is None or self.restyle_data is None:
        return
    self._figure.plotly_restyle(*self.restyle_data)