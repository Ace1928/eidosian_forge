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
def test_tabulator_sort_algorithm(page):
    df = pd.DataFrame({'vals': ['A', 'i', 'W', 'g', 'r', 'l', 'a', 'n', 'z', 'N', 'a', 'l', 's', 'm', 'J', 'C', 'w'], 'groups': ['A', 'B', 'C', 'B', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'A', 'A']})
    target_col = 'vals'
    widget = Tabulator(df, sorters=[{'field': 'groups', 'dir': 'asc'}])
    values = []
    widget.on_click(lambda e: values.append((e.column, e.row, e.value)))
    serve_component(page, widget)
    target_val = 'i'
    target_index = df.set_index(target_col).index.get_loc(target_val)
    cell = page.locator(f'text="{target_val}"')
    cell.click()
    wait_until(lambda: len(values) == 1, page)
    assert values[0] == (target_col, target_index, target_val)
    target_val = 'W'
    target_index = df.set_index(target_col).index.get_loc(target_val)
    cell = page.locator(f'text="{target_val}"')
    cell.click()
    wait_until(lambda: len(values) == 2, page)
    assert values[1] == (target_col, target_index, target_val)