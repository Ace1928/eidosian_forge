from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
@param.depends('relayout_data', watch=True)
def _update_figure_layout(self):
    if self._figure is None or self.relayout_data is None:
        return
    relayout_data = self._clean_relayout_data(self.relayout_data)
    self._figure.layout._compound_array_props.clear()
    self._figure.plotly_relayout(relayout_data)