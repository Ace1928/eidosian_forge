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
@classmethod
def _update_sources(cls, json_data, sources):
    layers = json_data.get('layers', [])
    source_columns = defaultdict(list)
    for i, source in enumerate(sources):
        key = tuple(sorted(source.data.keys()))
        source_columns[key].append((i, source))
    unprocessed, unused = ([], list(sources))
    for layer in layers:
        data = layer.get('data')
        if is_dataframe(data):
            data = ColumnDataSource.from_df(data)
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            data = cls._process_data(data)
        else:
            continue
        key = tuple(sorted(data.keys()))
        existing = source_columns.get(key)
        if existing:
            index, cds = existing.pop()
            layer['data'] = index
            updates = {}
            for col, values in data.items():
                if not np.array_equal(data[col], cds.data[col]):
                    updates[col] = values
            if updates:
                cds.data.update(updates)
            unused.remove(cds)
        else:
            unprocessed.append((layer, data))
    for layer, data in unprocessed:
        if unused:
            cds = unused.pop()
            cds.data = data
        else:
            cds = ColumnDataSource(data)
            sources.append(cds)
        layer['data'] = sources.index(cds)