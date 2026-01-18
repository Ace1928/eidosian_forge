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
def test_table_index_column(document, comm):
    df = pd.DataFrame({'int': [1, 2, 3], 'float': [3.14, 6.28, 9.42], 'index': ['A', 'B', 'C']}, index=[1, 2, 3])
    table = DataFrame(value=df)
    model = table.get_root(document, comm=comm)
    assert np.array_equal(model.source.data['level_0'], np.array([1, 2, 3]))
    assert model.columns[0].field == 'level_0'
    assert model.columns[0].title == ''