import asyncio
import os
import pandas as pd
import param
import pytest
from bokeh.models import (
from packaging.version import Version
from panel import config
from panel.depends import bind
from panel.io.state import set_curdoc, state
from panel.layout import Row, Tabs
from panel.models import HTML as BkHTML
from panel.pane import (
from panel.param import (
from panel.tests.util import mpl_available, mpl_figure
from panel.widgets import (
@mpl_available
def test_param_method_pane_changing_type(document, comm):
    test = View()
    pane = panel(test.mixed_view)
    inner_pane = pane._pane
    assert isinstance(inner_pane, Matplotlib)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkColumn)
    assert len(row.children) == 1
    model = row.children[0]
    text = model.text
    assert text.startswith('&lt;img src=')
    test.a = 5
    new_model = row.children[0]
    new_pane = pane._pane
    assert isinstance(new_pane, Bokeh)
    assert isinstance(new_model, Div)
    assert new_model.text != text
    new_pane._cleanup(row)
    assert new_pane._models == {}