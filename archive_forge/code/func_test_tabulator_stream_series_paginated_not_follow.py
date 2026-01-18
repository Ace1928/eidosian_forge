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
def test_tabulator_stream_series_paginated_not_follow(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, pagination='remote', page_size=2)
    model = table.get_root(document, comm)
    stream_value = pd.Series({'A': 5, 'B': 1, 'C': 'foo6', 'D': dt.datetime(2009, 1, 8)})
    table.stream(stream_value, follow=False)
    assert table.page == 1
    assert len(table.value) == 6
    expected = {'index': np.array([0, 1]), 'A': np.array([0, 1]), 'B': np.array([0, 1]), 'C': np.array(['foo1', 'foo2']), 'D': np.array(['2009-01-01T00:00:00.000000000', '2009-01-02T00:00:00.000000000'], dtype='datetime64[ns]').astype(np.int64) / 1000000.0}
    for col, values in model.source.data.items():
        np.testing.assert_array_equal(values, expected[col])