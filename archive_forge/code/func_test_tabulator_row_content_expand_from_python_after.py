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
def test_tabulator_row_content_expand_from_python_after(page, df_mixed):
    widget = Tabulator(df_mixed, row_content=lambda i: f'{i['str']}-row-content')
    serve_component(page, widget)
    widget.expanded = [0, 2]
    for i in range(len(df_mixed)):
        row_content = page.locator(f'text="{df_mixed.iloc[i]['str']}-row-content"')
        if i in widget.expanded:
            expect(row_content).to_have_count(1)
        else:
            expect(row_content).to_have_count(0)
    expect(page.locator('text="▼"')).to_have_count(len(widget.expanded))
    expect(page.locator('text="►"')).to_have_count(len(df_mixed) - len(widget.expanded))
    widget.expanded = []
    expect(page.locator('text="▼"')).to_have_count(0)
    expect(page.locator('text="►"')).to_have_count(len(df_mixed))