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
def test_tabulator_click_event_and_header_filters(page):
    df = pd.DataFrame({'col1': list('ABCDD'), 'col2': list('XXXXZ')})
    widget = Tabulator(df, header_filters={'col1': {'type': 'input', 'func': 'like'}})
    values = []
    widget.on_click(lambda e: values.append((e.column, e.row, e.value)))
    serve_component(page, widget)
    str_header = page.locator('input[type="search"]')
    str_header.click()
    str_header.fill('D')
    str_header.press('Enter')
    wait_until(lambda: len(widget.filters) == 1, page)
    cell = page.locator('text="Z"')
    cell.click()
    wait_until(lambda: len(values) == 1, page)
    assert values[0] == ('col2', 4, 'Z')