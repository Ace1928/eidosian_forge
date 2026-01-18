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
def test_tabulator_streaming_default(page):
    df = pd.DataFrame(np.random.random((3, 2)), columns=['A', 'B'])
    widget = Tabulator(df)
    serve_component(page, widget)
    expect(page.locator('.tabulator-row')).to_have_count(len(df))
    height_start = page.locator('.pnx-tabulator.tabulator').bounding_box()['height']

    def stream_data():
        widget.stream(df)
    repetitions = 3
    state.add_periodic_callback(stream_data, period=100, count=repetitions)
    expected_len = len(df) * (repetitions + 1)
    expect(page.locator('.tabulator-row')).to_have_count(expected_len)
    assert len(widget.value) == expected_len
    assert widget.current_view.equals(widget.value)
    assert page.locator('.pnx-tabulator.tabulator').bounding_box()['height'] > height_start