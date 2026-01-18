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
def test_tabulator_patch_event():
    df = makeMixedDataFrame()
    table = Tabulator(df)
    values = []
    table.on_edit(lambda e: values.append((e.column, e.row, e.value)))
    for col in df.columns:
        for row in range(len(df)):
            event = TableEditEvent(model=None, column=col, row=row)
            table._process_event(event)
            assert values[-1] == (col, row, df[col].iloc[row])