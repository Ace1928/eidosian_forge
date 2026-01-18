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
@pytest.mark.parametrize('pagination', (pytest.param('local', marks=pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3553')), pytest.param('remote', marks=pytest.mark.xfail(reason='See https://github.com/holoviz/panel/issues/3553')), None))
def test_tabulator_header_filter_no_horizontal_rescroll(page, df_mixed, pagination):
    widths = 100
    width = int((df_mixed.shape[1] + 1) * widths / 2)
    col_name = 'newcol'
    df_mixed[col_name] = 'on'
    widget = Tabulator(df_mixed, width=width, widths=widths, header_filters={col_name: {'type': 'input', 'func': 'like'}}, pagination=pagination)
    serve_component(page, widget)
    header = page.locator(f'text="{col_name}"')
    header.scroll_into_view_if_needed()
    bb = header.bounding_box()
    header = page.locator('input[type="search"]')
    header.click()
    header.fill('off')
    header.press('Enter')
    page.wait_for_timeout(400)
    assert page.locator(f'text="{col_name}"').bounding_box() == bb