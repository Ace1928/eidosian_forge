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
def test_tabulator_formatters_bokeh_scientific(page, df_mixed):
    df_mixed['float'] = df_mixed['float'] * 1000000.0
    df_mixed.loc['idx1', 'float'] = np.nan
    widget = Tabulator(df_mixed, formatters={'float': ScientificFormatter(precision=3, nan_format='nan-float')})
    serve_component(page, widget)
    expect(page.locator('text="3.140e+6"')).to_have_count(1)
    assert page.locator('text="nan-float"').count() == 1