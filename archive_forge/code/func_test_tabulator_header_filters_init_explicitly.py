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
def test_tabulator_header_filters_init_explicitly(page, df_mixed):
    header_filters = {'float': {'type': 'number', 'func': '>=', 'placeholder': 'Placeholder float'}, 'str': {'type': 'input', 'func': 'like', 'placeholder': 'Placeholder str'}}
    widget = Tabulator(df_mixed, header_filters=header_filters)
    serve_component(page, widget)
    expect(page.locator('.tabulator-header-filter')).to_have_count(len(header_filters))
    number_header = page.locator('input[type="number"]')
    expect(number_header).to_have_count(1)
    assert number_header.get_attribute('placeholder') == 'Placeholder float'
    str_header = page.locator('input[type="search"]')
    expect(str_header).to_have_count(1)
    assert str_header.get_attribute('placeholder') == 'Placeholder str'