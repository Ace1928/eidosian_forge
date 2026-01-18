from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
def x_values():
    try:
        table_values = [int(v) for v in tabulator_column_values(page, 'x')]
    except Exception:
        return False
    if table_values:
        assert table_values == list(df2['x'].sort_values(ascending=False))
    else:
        return False