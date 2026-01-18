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
def test_tabulator_selection_default_selection(page, df_mixed):
    selection = [0, 2]
    widget = Tabulator(df_mixed, selection=[0, 2])
    serve_component(page, widget)
    rows = page.locator('.tabulator-row')
    for i in range(rows.count()):
        if i in selection:
            assert 'tabulator-selected' in rows.nth(i).get_attribute('class')
        else:
            assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')
    expected_selected = df_mixed.loc[['idx0', 'idx2'], :]
    assert widget.selected_dataframe.equals(expected_selected)