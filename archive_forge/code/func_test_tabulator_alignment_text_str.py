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
def test_tabulator_alignment_text_str(page, df_mixed):
    talign = 'center'
    widget = Tabulator(df_mixed, text_align=talign)
    serve_component(page, widget)
    cells = page.locator('.tabulator-cell:visible')
    expect(cells).to_have_count(len(df_mixed) * (df_mixed.shape[1] + 1))
    for i in range(cells.count()):
        expect(cells.nth(i)).to_have_css('text-align', talign)