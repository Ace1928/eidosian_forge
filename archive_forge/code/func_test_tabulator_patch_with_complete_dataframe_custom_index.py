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
def test_tabulator_patch_with_complete_dataframe_custom_index(document, comm):
    df = makeMixedDataFrame()[['A', 'B', 'C']]
    df.index = [0, 1, 2, 3, 10]
    table = Tabulator(df)
    model = table.get_root(document, comm)
    table.patch(df)
    expected = {'index': np.array([0, 1, 2, 3, 10]), 'A': np.array([0, 1, 2, 3, 4]), 'B': np.array([0, 1, 0, 1, 0]), 'C': np.array(['foo1', 'foo2', 'foo3', 'foo4', 'foo5'])}
    for col, values in model.source.data.items():
        expected_array = expected[col]
        np.testing.assert_array_equal(values, expected_array)
        if col != 'index':
            np.testing.assert_array_equal(table.value[col].values, expected[col])