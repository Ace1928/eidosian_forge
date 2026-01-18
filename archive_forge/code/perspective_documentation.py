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

        Register a callback to be executed when any row is clicked.
        The callback is given a PerspectiveClickEvent declaring the
        config, column names, and row values of the row that was
        clicked.

        Arguments
        ---------
        callback: (callable)
            The callback to run on edit events.
        