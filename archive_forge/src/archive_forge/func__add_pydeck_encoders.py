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
def _add_pydeck_encoders(cls):
    if cls._pydeck_encoders_are_added or 'pydeck' not in sys.modules:
        return
    from pydeck.types import String

    def pydeck_string_encoder(obj, serializer):
        return obj.value
    Serializer._encoders[String] = pydeck_string_encoder