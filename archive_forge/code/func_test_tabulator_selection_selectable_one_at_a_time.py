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
def test_tabulator_selection_selectable_one_at_a_time(page, df_mixed):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    rows = page.locator('.tabulator-row')
    c0 = page.locator('text="idx0"')
    c0.wait_for()
    c0.click()
    wait_until(lambda: widget.selection == [0], page)
    expected_selected = df_mixed.loc[['idx0'], :]
    assert widget.selected_dataframe.equals(expected_selected)
    page.locator('text="idx1"').click()
    wait_until(lambda: widget.selection == [1], page)
    expected_selected = df_mixed.loc[['idx1'], :]
    assert widget.selected_dataframe.equals(expected_selected)
    for i in range(rows.count()):
        if i == 1:
            assert 'tabulator-selected' in rows.nth(i).get_attribute('class')
        else:
            assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')
    page.locator('text="idx1"').click()
    wait_until(lambda: widget.selection == [1], page)
    assert widget.selected_dataframe.equals(expected_selected)
    for i in range(rows.count()):
        if i == 1:
            assert 'tabulator-selected' in rows.nth(i).get_attribute('class')
        else:
            assert 'tabulator-selected' not in rows.nth(i).get_attribute('class')