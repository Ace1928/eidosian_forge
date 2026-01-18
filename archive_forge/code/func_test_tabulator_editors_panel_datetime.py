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
def test_tabulator_editors_panel_datetime(page, df_mixed):
    widget = Tabulator(df_mixed, editors={'datetime': 'datetime'})
    serve_component(page, widget)
    cell = page.locator('text="2019-01-01 10:00:00"')
    cell.click()
    cell_edit = page.locator('input[type="datetime-local"]')
    new_datetime = dt.datetime(1980, 11, 30, 4, 51, 0)
    time_to_fill = new_datetime.isoformat()
    time_to_fill = time_to_fill[:-3]
    cell_edit.fill(time_to_fill)
    page.locator('input[type="datetime-local"]').press('Enter')
    new_datetime_display = new_datetime.strftime('%Y-%m-%d %H:%M:%S')
    expect(page.locator(f'text="{new_datetime_display}"')).to_have_count(1)
    wait_until(lambda: new_datetime in widget.value['datetime'].tolist(), page)
    cell = page.locator(f'text="{new_datetime_display}"')
    cell.click()
    cell_edit = page.locator('input[type="datetime-local"]')
    new_datetime2 = dt.datetime(1990, 3, 31, 12, 45, 0)
    time_to_fill2 = new_datetime2.isoformat()
    time_to_fill2 = time_to_fill2[:-3]
    cell_edit.fill(time_to_fill2)
    page.locator('input[type="datetime-local"]').press('Escape')
    new_datetime_display2 = new_datetime2.strftime('%Y-%m-%d %H:%M:%S')
    expect(page.locator(f'text="{new_datetime_display2}"')).to_have_count(0)
    assert new_datetime2 not in widget.value['datetime'].tolist()