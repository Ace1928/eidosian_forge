from __future__ import annotations
import datetime as dt
import sys
from enum import Enum
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import ColumnDataSource, ImportedStyleSheet
from pyviz_comms import JupyterComm
from ..io.state import state
from ..reactive import ReactiveData
from ..util import datetime_types, lazy_load
from ..viewable import Viewable
from .base import ModelPane
def _filter_properties(self, properties):
    ignored = list(Viewable.param)
    return [p for p in properties if p not in ignored]