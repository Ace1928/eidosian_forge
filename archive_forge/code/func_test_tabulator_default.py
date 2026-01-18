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
def test_tabulator_default(page, df_mixed, df_mixed_as_string):
    nrows, ncols = df_mixed.shape
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    expected_ncols = ncols + 2
    table = page.locator('.pnx-tabulator.tabulator')
    expect(table).to_have_text(df_mixed_as_string, use_inner_text=True)
    assert widget.layout == 'fit_data_table'
    assert table.get_attribute('tabulator-layout') == 'fitDataTable'
    rows = page.locator('.tabulator-row')
    assert rows.count() == nrows
    cols = page.locator('.tabulator-col')
    assert cols.count() == expected_ncols
    assert cols.nth(0).get_attribute('tabulator-field') == '_index'
    assert cols.nth(0).is_hidden()
    assert widget.show_index
    assert page.locator('text="index"').is_visible()
    assert cols.nth(1).is_visible()
    assert page.locator('.tabulator-sortable').count() == expected_ncols
    for i in range(expected_ncols):
        assert cols.nth(i).get_attribute('aria-sort') == 'none'