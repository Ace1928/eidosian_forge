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
def test_tabulator_numeric_groups(document, comm):
    df = pd.DataFrame(np.random.rand(10, 3))
    table = Tabulator(df, groups={'Number': [0, 1]})
    model = table.get_root(document, comm)
    assert model.configuration['columns'] == [{'field': 'index', 'sorter': 'number'}, {'title': 'Number', 'columns': [{'field': '0', 'sorter': 'number'}, {'field': '1', 'sorter': 'number'}]}, {'field': '2', 'sorter': 'number'}]