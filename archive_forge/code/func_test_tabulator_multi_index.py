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
def test_tabulator_multi_index(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df.set_index(['A', 'C']))
    model = table.get_root(document, comm)
    assert model.configuration['columns'] == [{'field': 'A', 'sorter': 'number'}, {'field': 'C'}, {'field': 'B', 'sorter': 'number'}, {'field': 'D', 'sorter': 'timestamp'}]
    assert np.array_equal(model.source.data['A'], np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.array_equal(model.source.data['C'], np.array(['foo1', 'foo2', 'foo3', 'foo4', 'foo5']))