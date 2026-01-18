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
def test_tabulator_pagination_remote_cell_click_event():
    df = makeMixedDataFrame()
    table = Tabulator(df, pagination='remote', page_size=2)
    values = []
    table.on_click(lambda e: values.append((e.column, e.row, e.value)))
    data = df.reset_index()
    for col in data.columns:
        for p in range(len(df) // 2):
            table.page = p + 1
            for row in range(2):
                event = CellClickEvent(model=None, column=col, row=row)
                table._process_event(event)
                assert values[-1] == (col, p * 2 + row, data[col].iloc[p * 2 + row])