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
def test_tabulator_sorters_unnamed_index(document, comm):
    df = pd.DataFrame(np.random.rand(10, 4))
    table = Tabulator(df)
    table.sorters = [{'field': 'index', 'sorter': 'number', 'dir': 'desc'}]
    pd.testing.assert_frame_equal(table.current_view, df.sort_index(ascending=False))