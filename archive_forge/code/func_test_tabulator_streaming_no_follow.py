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
def test_tabulator_streaming_no_follow(page):
    nrows1 = 10
    arr = np.random.randint(10, 20, (nrows1, 2))
    val = [-1]
    arr[0, :] = val[0]
    df = pd.DataFrame(arr, columns=['A', 'B'])
    widget = Tabulator(df, height=100)
    serve_component(page, widget)
    expect(page.locator('.tabulator-row')).to_have_count(len(df))
    assert page.locator('text="-1"').count() == 2
    height_start = page.locator('.pnx-tabulator.tabulator').bounding_box()['height']
    recs = []
    nrows2 = 5

    def stream_data():
        arr = np.random.randint(10, 20, (nrows2, 2))
        val[0] = val[0] - 1
        arr[-1, :] = val[0]
        recs.append(val[0])
        new_df = pd.DataFrame(arr, columns=['A', 'B'])
        widget.stream(new_df, follow=False)
    repetitions = 3
    state.add_periodic_callback(stream_data, period=100, count=repetitions)
    page.wait_for_timeout(500)
    expect(page.locator('text="-1"')).to_have_count(2)
    expect(page.locator(f'text="{val[0]}"')).to_have_count(0)
    assert len(widget.value) == nrows1 + repetitions * nrows2
    assert widget.current_view.equals(widget.value)
    assert page.locator('.pnx-tabulator.tabulator').bounding_box()['height'] == height_start