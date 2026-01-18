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
def test_tabulator_groups(page, df_mixed):
    widget = Tabulator(df_mixed, groups={'Group1': ['int', 'float'], 'Group2': ['date', 'datetime']})
    serve_component(page, widget)
    expected_text = '\n    index\n    Group1\n    int\n    float\n    str\n    bool\n    Group2\n    date\n    datetime\n    idx0\n    1\n    3.14\n    A\n    true\n    2019-01-01\n    2019-01-01 10:00:00\n    idx1\n    2\n    6.28\n    B\n    true\n    2020-01-01\n    2020-01-01 12:00:00\n    idx2\n    3\n    9.42\n    C\n    true\n    2020-01-10\n    2020-01-10 13:00:00\n    idx3\n    4\n    -2.45\n    D\n    false\n    2019-01-10\n    2020-01-15 13:00:00\n    '
    expect(page.locator('.tabulator')).to_have_text(expected_text, use_inner_text=True)
    expect(page.locator('.tabulator-col-group')).to_have_count(2)