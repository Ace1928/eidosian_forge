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
def test_tabulator_patch_with_dataframe_custom_index_multiple_error(document, comm):
    df = pd.DataFrame(dict(A=[1, 4, 2]), index=['foo1', 'foo1', 'foo3'])
    original = df.copy()
    df_patch = pd.DataFrame(dict(A=[20, 10]), index=['foo1', 'foo1'])
    table = Tabulator(df)
    with pytest.raises(ValueError, match="Patching a table with duplicate index values is not supported\\. Found this duplicate index: 'foo1'"):
        table.patch(df_patch)
    pd.testing.assert_frame_equal(table.value, original)