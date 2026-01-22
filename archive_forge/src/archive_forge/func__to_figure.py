from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
def _to_figure(self, obj):
    import plotly.graph_objs as go
    if isinstance(obj, go.Figure):
        return obj
    elif isinstance(obj, dict):
        data, layout = (obj['data'], obj['layout'])
    elif isinstance(obj, tuple):
        data, layout = obj
    else:
        data, layout = (obj, {})
    data = data if isinstance(data, list) else [data]
    return go.Figure(data=data, layout=layout)