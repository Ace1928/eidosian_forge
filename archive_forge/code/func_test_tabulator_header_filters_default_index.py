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
@pytest.mark.parametrize(('index', 'expected_selector'), ((['idx0', 'idx1'], 'input[type="search"]'), ([0, 1], 'input[type="number"]'), (np.array([0, 1], dtype=np.uint64), 'input[type="number"]'), ([0.1, 1.1], 'input[type="number"]')))
def test_tabulator_header_filters_default_index(page, index, expected_selector):
    df = pd.DataFrame(index=index)
    widget = Tabulator(df, header_filters=True)
    serve_component(page, widget)
    expect(page.locator(expected_selector)).to_have_count(1)