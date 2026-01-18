import asyncio
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models.widgets.tables import (
from packaging.version import Version
from panel.depends import bind
from panel.io.state import set_curdoc
from panel.models.tabulator import CellClickEvent, TableEditEvent
from panel.tests.util import mpl_available, serve_and_request, wait_until
from panel.util import BOKEH_JS_NAT
from panel.widgets import Button, TextInput
from panel.widgets.tables import DataFrame, Tabulator
def test_tabulator_formatters_default(document, comm):
    df = pd.DataFrame({'int': [1, 2], 'float': [3.14, 6.28], 'str': ['A', 'B'], 'date': [dt.date(2009, 1, 8), dt.date(2010, 1, 8)], 'datetime': [dt.datetime(2009, 1, 8), dt.datetime(2010, 1, 8)]})
    table = Tabulator(df)
    model = table.get_root(document, comm)
    mformatter = model.columns[1].formatter
    assert isinstance(mformatter, NumberFormatter)
    mformatter = model.columns[2].formatter
    assert isinstance(mformatter, NumberFormatter)
    assert mformatter.format == '0,0.0[00000]'
    mformatter = model.columns[3].formatter
    assert isinstance(mformatter, StringFormatter)
    mformatter = model.columns[4].formatter
    assert isinstance(mformatter, DateFormatter)
    assert mformatter.format == '%Y-%m-%d'
    mformatter = model.columns[5].formatter
    assert isinstance(mformatter, DateFormatter)
    assert mformatter.format == '%Y-%m-%d %H:%M:%S'