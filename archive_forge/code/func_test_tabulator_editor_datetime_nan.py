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
@pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3620')
def test_tabulator_editor_datetime_nan(page, df_mixed):
    df_mixed.at['idx0', 'datetime'] = np.nan
    widget = Tabulator(df_mixed, configuration={'headerSort': False})
    events = []

    def callback(e):
        events.append(e)
    widget.on_edit(callback)
    serve_component(page, widget)
    cell = page.locator('text="-"')
    cell.wait_for()
    cell.click()
    page.locator('input[type="date"]').press('Escape')
    page.locator('text="-"').click()
    page.locator('input[type="date"]').press('Enter')
    page.locator('text="-"').click()
    page.locator('html').click()
    wait_until(lambda: len(events) == 0, page)