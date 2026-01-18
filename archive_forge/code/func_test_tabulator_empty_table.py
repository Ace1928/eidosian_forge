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
def test_tabulator_empty_table(document, comm):
    value_df = makeMixedDataFrame()
    empty_df = pd.DataFrame([], columns=value_df.columns)
    table = Tabulator(empty_df)
    table.get_root(document, comm)
    assert table.value.shape == empty_df.shape
    table.stream(value_df, follow=True)
    assert table.value.shape == value_df.shape