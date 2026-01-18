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
def test_tabulator_expanded_content_embed(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, embed_content=True, row_content=lambda r: r.A)
    model = table.get_root(document, comm)
    assert len(model.children) == len(df)
    for i, r in df.iterrows():
        assert i in model.children
        row = model.children[i]
        assert row.text == f'&lt;pre&gt;{r.A}&lt;/pre&gt;'
    table.row_content = lambda r: r.A + 1
    for i, r in df.iterrows():
        assert i in model.children
        row = model.children[i]
        assert row.text == f'&lt;pre&gt;{r.A + 1}&lt;/pre&gt;'