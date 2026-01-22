from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import numpy as np
import param
from bokeh.core.serialization import Serializer
from bokeh.models import ColumnDataSource
from pyviz_comms import JupyterComm
from ..util import is_dataframe, lazy_load
from .base import ModelPane
def _transform_deck_object(self, obj):
    data = dict(obj.__dict__)
    mapbox_api_key = data.pop('mapbox_key', '') or self.mapbox_api_key
    deck_widget = data.pop('deck_widget', None)
    if isinstance(self.tooltips, dict) or deck_widget is None:
        tooltip = self.tooltips
    else:
        tooltip = deck_widget.tooltip
    data = {k: v for k, v in recurse_data(data).items() if v is not None}
    if 'initialViewState' in data:
        data['initialViewState'] = {k: v for k, v in data['initialViewState'].items() if v is not None}
    self._add_pydeck_encoders()
    return (data, tooltip, mapbox_api_key)