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
def test_tabulator_hidden_columns(page, df_mixed):
    widget = Tabulator(df_mixed, hidden_columns=['float', 'date', 'datetime'])
    serve_component(page, widget)
    expected_text = '\n        index\n        int\n        str\n        bool\n        idx0\n        1\n        A\n        true\n        idx1\n        2\n        B\n        true\n        idx2\n        3\n        C\n        true\n        idx3\n        4\n        D\n        false\n    '
    table = page.locator('.pnx-tabulator.tabulator')
    expect(table).to_have_text(expected_text, use_inner_text=True)