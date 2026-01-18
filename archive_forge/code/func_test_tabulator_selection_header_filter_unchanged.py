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
@pytest.mark.xfail(reason='https://github.com/holoviz/panel/issues/3664')
def test_tabulator_selection_header_filter_unchanged(page):
    df = pd.DataFrame({'col1': list('XYYYYY'), 'col2': list('abcddd'), 'col3': list('ABCDEF')})
    selection = [2, 3]
    widget = Tabulator(df, selection=selection, header_filters={'col1': {'type': 'input', 'func': 'like'}})
    serve_component(page, widget)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    str_header.fill('Y')
    str_header.press('Enter')
    page.wait_for_timeout(300)
    assert widget.selection == selection
    expected_selected = df.iloc[selection, :]
    assert widget.selected_dataframe.equals(expected_selected)