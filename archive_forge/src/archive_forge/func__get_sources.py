from __future__ import annotations
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import isdatetime, lazy_load
from ..viewable import Layoutable
from .base import ModelPane
@staticmethod
def _get_sources(json):
    sources = []
    traces = json.get('data', [])
    for trace in traces:
        data = {}
        Plotly._get_sources_for_trace(trace, data)
        sources.append(ColumnDataSource(data))
    return sources