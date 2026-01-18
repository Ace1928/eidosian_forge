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
def test_tabulator_stream_dataframe_selectable_rows(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, selectable_rows=lambda df: list(range(0, len(df), 2)))
    model = table.get_root(document, comm)
    assert model.selectable_rows == [0, 2, 4]
    stream_value = pd.DataFrame({'A': [5, 6], 'B': [1, 0], 'C': ['foo6', 'foo7'], 'D': [dt.datetime(2009, 1, 8), dt.datetime(2009, 1, 9)]})
    table.stream(stream_value)
    assert model.selectable_rows == [0, 2, 4, 6]