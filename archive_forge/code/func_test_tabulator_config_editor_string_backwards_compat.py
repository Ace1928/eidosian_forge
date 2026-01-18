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
def test_tabulator_config_editor_string_backwards_compat(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, editors={'B': 'select'})
    model = table.get_root(document, comm)
    assert model.configuration['columns'][2] == {'field': 'B', 'sorter': 'number', 'editor': 'list'}