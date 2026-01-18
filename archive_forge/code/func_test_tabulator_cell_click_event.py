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
def test_tabulator_cell_click_event(page, df_mixed):
    widget = Tabulator(df_mixed)
    values = []
    widget.on_click(lambda e: values.append((e.column, e.row, e.value)))
    serve_component(page, widget)
    page.locator('text="idx0"').click()
    wait_until(lambda: len(values) >= 1, page)
    assert values[-1] == ('index', 0, 'idx0')
    page.locator('text="A"').click()
    wait_until(lambda: len(values) >= 2, page)
    assert values[-1] == ('str', 0, 'A')