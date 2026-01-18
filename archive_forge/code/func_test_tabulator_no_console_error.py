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
def test_tabulator_no_console_error(page, df_mixed):
    widget = Tabulator(df_mixed)
    msgs, _ = serve_component(page, widget)
    page.wait_for_timeout(1000)
    assert [msg for msg in msgs if msg.type == 'error' and 'favicon' not in msg.location['url']] == []