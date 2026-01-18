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
@pytest.mark.parametrize('editor', ['select', 'autocomplete'])
def test_tabulator_header_filters_column_config_select_autocomplete_backwards_compat(document, comm, editor):
    df = makeMixedDataFrame()
    table = Tabulator(df, header_filters={'C': editor, 'D': {'type': editor, 'values': True}})
    model = table.get_root(document, comm)
    assert model.configuration['columns'] == [{'field': 'index', 'sorter': 'number'}, {'field': 'A', 'sorter': 'number'}, {'field': 'B', 'sorter': 'number'}, {'field': 'C', 'headerFilter': 'list', 'headerFilterParams': {'valuesLookup': True}}, {'field': 'D', 'headerFilter': 'list', 'headerFilterParams': {'valuesLookup': True}, 'sorter': 'timestamp'}]
    assert model.configuration['selectable'] == True