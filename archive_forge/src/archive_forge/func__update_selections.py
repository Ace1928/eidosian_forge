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
def _update_selections(self, *args):
    params = {e: param.Dict(allow_refs=False) if stype == 'interval' else param.List(allow_refs=False) for e, stype in self._selections.items()}
    if self.selection and set(self.selection.param) - {'name'} == set(params):
        self.selection.param.update({p: None for p in params})
        return
    self.selection = type('Selection', (param.Parameterized,), params)()