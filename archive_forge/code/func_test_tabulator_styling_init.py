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
def test_tabulator_styling_init(page, df_mixed):
    df_styled = df_mixed.style.apply(highlight_max, subset=['int']).applymap(color_false, subset=['bool'])
    widget = Tabulator(df_styled)
    serve_component(page, widget)
    max_int = df_mixed['int'].max()
    max_cell = page.locator('.tabulator-cell', has=page.locator(f'text="{max_int}"'))
    expect(max_cell).to_have_count(1)
    expect(max_cell).to_have_css('background-color', _color_mapping['yellow'])
    expect(page.locator('text="false"')).to_have_css('color', _color_mapping['red'])