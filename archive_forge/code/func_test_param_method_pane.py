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
def test_param_method_pane(document, comm):
    test = View()
    pane = panel(test.view)
    inner_pane = pane._pane
    assert isinstance(inner_pane, Bokeh)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkColumn)
    assert len(row.children) == 1
    model = row.children[0]
    assert pane._models[row.ref['id']][0] is row
    assert isinstance(model, Div)
    assert model.text == '0'
    test.a = 5
    new_model = row.children[0]
    assert inner_pane is pane._pane
    assert new_model.text == '5'
    assert pane._models[row.ref['id']][0] is row
    pane._cleanup(row)
    assert pane._models == {}
    assert inner_pane._models == {}