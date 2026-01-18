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
def test_tabulator_patch_no_height_resize(page):
    header = Column('Text', height=1000)
    df = pd.DataFrame(np.random.random((150, 1)), columns=['a'])
    widget = Tabulator(df)
    app = Column(header, widget)
    serve_component(page, app)
    page.mouse.wheel(delta_x=0, delta_y=10000)
    at_bottom_script = '\n    isAtBottom => (window.innerHeight + window.scrollY) >= document.body.scrollHeight;\n    '
    wait_until(lambda: page.evaluate(at_bottom_script), page)
    widget.patch({'a': [(len(df) - 1, 100)]})
    page.wait_for_timeout(400)
    wait_until(lambda: page.evaluate(at_bottom_script), page)