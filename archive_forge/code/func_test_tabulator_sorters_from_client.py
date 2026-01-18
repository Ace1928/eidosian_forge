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
def test_tabulator_sorters_from_client(page, df_mixed):
    widget = Tabulator(df_mixed)
    serve_component(page, widget)
    page.locator('.tabulator-col', has_text='float').locator('.tabulator-col-sorter').click()
    sheader = page.locator('[aria-sort="ascending"]:visible')
    expect(sheader).to_have_count(1)
    assert sheader.get_attribute('tabulator-field') == 'float'
    wait_until(lambda: widget.sorters == [{'field': 'float', 'dir': 'asc'}], page)
    expected_df_sorted = df_mixed.sort_values('float', ascending=True)
    assert widget.current_view.equals(expected_df_sorted)