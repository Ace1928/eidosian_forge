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
def test_tabulator_selected_and_filtered_dataframe(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, selection=list(range(len(df))))
    pd.testing.assert_frame_equal(table.selected_dataframe, df)
    table.add_filter('foo3', 'C')
    pd.testing.assert_frame_equal(table.selected_dataframe, df[df['C'] == 'foo3'])
    table.remove_filter('foo3')
    table.selection = [0, 1, 2]
    table.add_filter('foo3', 'C')
    assert table.selection == [0]