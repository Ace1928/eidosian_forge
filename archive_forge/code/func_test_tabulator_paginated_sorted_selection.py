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
def test_tabulator_paginated_sorted_selection(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, pagination='remote', page_size=2)
    table.sorters = [{'field': 'A', 'sorter': 'number', 'dir': 'dec'}]
    model = table.get_root(document, comm)
    table.selection = [3]
    assert model.source.selected.indices == [1]
    table.selection = [0, 1]
    assert model.source.selected.indices == []
    table.selection = [3, 4]
    assert model.source.selected.indices == [1, 0]
    table.selection = []
    assert model.source.selected.indices == []
    table._process_events({'indices': [0, 1]})
    assert table.selection == [4, 3]
    table._process_events({'indices': [1]})
    assert table.selection == [3]
    table.sorters = [{'field': 'A', 'sorter': 'number', 'dir': 'asc'}]
    table._process_events({'indices': [1]})
    assert table.selection == [1]