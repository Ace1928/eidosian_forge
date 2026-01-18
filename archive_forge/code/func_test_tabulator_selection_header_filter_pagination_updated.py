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
@pytest.mark.parametrize('pagination', ['remote', 'local'])
def test_tabulator_selection_header_filter_pagination_updated(page, df_mixed, pagination):
    widget = Tabulator(df_mixed, header_filters={'str': {'type': 'input', 'func': 'like'}}, pagination=pagination, page_size=3)
    serve_component(page, widget)
    page.locator('text="Last"').click()
    wait_until(lambda: widget.page == 2, page)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    str_header.fill('D')
    str_header.press('Enter')
    wait_until(lambda: widget.page == 1, page)