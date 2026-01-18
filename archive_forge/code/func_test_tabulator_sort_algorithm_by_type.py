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
@pytest.mark.parametrize(('col', 'vals'), (('string', [np.nan, '', 'B', 'a', '', np.nan]), ('number', [1.0, 1.0, 0.0, 0.0]), ('boolean', [True, True, False, False]), ('datetime', [dt.datetime(2019, 1, 1, 1), np.nan, dt.datetime(2019, 12, 1, 1), dt.datetime(2019, 12, 1, 1), np.nan, dt.datetime(2019, 6, 1, 1), np.nan])))
def test_tabulator_sort_algorithm_by_type(page, col, vals):
    df = pd.DataFrame({col: vals})
    widget = Tabulator(df, sorters=[{'field': col, 'dir': 'asc'}])
    serve_component(page, widget)
    page.wait_for_timeout(200)
    client_index = [int(i) for i in tabulator_column_values(page, 'index')]

    def indexes_equal():
        assert client_index == list(widget.current_view.index)
    wait_until(indexes_equal, page)