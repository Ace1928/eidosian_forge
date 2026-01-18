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
@pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3657')
def test_tabulator_sorters_on_init_multiple(page):
    df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': [1, 4, 3, 2]})
    sorters = [{'field': 'col1', 'dir': 'desc'}, {'field': 'col2', 'dir': 'asc'}]
    widget = Tabulator(df, sorters=sorters)
    serve_component(page, widget)
    s1 = page.locator('[aria-sort="descending"]:visible')
    expect(s1).to_have_attribute('tabulator-field', 'col1')
    s2 = page.locator('[aria-sort="ascending"]:visible')
    expect(s2).to_have_attribute('tabulator-field', 'col2')
    first_index_rendered = page.locator('.tabulator-cell:visible').first.inner_text()
    df_sorted = df.sort_values('col1', ascending=True).sort_values('col2', ascending=False)
    expected_first_index = df_sorted.index[0]
    assert int(first_index_rendered) == expected_first_index