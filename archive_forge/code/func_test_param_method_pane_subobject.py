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
def test_param_method_pane_subobject(document, comm):
    subobject = View(name='Nested', a=42)
    test = View(b=subobject)
    pane = panel(test.subobject_view)
    inner_pane = pane._pane
    assert isinstance(inner_pane, Bokeh)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkColumn)
    assert len(row.children) == 1
    model = row.children[0]
    assert isinstance(model, Div)
    assert model.text == '42'
    watchers = pane._internal_callbacks
    assert any((w.inst is subobject for w in watchers))
    assert pane._models[row.ref['id']][0] is row
    new_subobject = View(name='Nested', a=42)
    test.b = new_subobject
    assert pane._models[row.ref['id']][0] is row
    watchers = pane._internal_callbacks
    assert not any((w.inst is subobject for w in watchers))
    assert any((w.inst is new_subobject for w in watchers))
    pane._cleanup(row)
    assert pane._models == {}
    assert inner_pane._models == {}