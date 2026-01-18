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
def test_tabulator_patch_with_timestamp(document, comm):
    df = pd.DataFrame(dict(A=pd.to_datetime(['1980-01-01', '1980-01-02'])))
    table = Tabulator(df)
    model = table.get_root(document, comm)
    table.patch({'A': [(0, pd.Timestamp('2021-01-01'))]})
    expected = {'index': np.array([0, 1]), 'A': np.array(['2021-01-01T00:00:00.000000000', '1980-01-02T00:00:00.000000000'], dtype='datetime64[ns]')}
    for col, values in model.source.data.items():
        if col == 'A':
            expected_array = expected[col].astype(np.int64) / 1000000.0
        else:
            expected_array = expected[col]
        np.testing.assert_array_equal(values, expected_array)
        if col != 'index':
            np.testing.assert_array_equal(table.value[col].values, expected[col])