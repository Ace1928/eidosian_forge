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
def test_set_widgets(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1, bounds=(0, 10), precedence=1)
        b = param.String(default='A', precedence=2)
    pane = Param(Test())
    model = pane.get_root(document, comm=comm)
    assert len(model.children) == 3
    title, slider, text = model.children
    assert isinstance(title, Div)
    assert isinstance(slider, Slider)
    assert isinstance(text, TextInput)
    pane.widgets = {'a': LiteralInput(value=1, type=(float, int))}
    assert len(model.children) == 3
    title, number, text = model.children
    assert isinstance(title, Div)
    assert isinstance(number, TextInput)
    assert isinstance(text, TextInput)
    pane.widgets = {'a': {'height': 100}}
    assert len(model.children) == 3
    title, number, text = model.children
    assert isinstance(title, Div)
    assert isinstance(number, Slider)
    assert number.height == 100
    assert isinstance(text, TextInput)
    pane.widgets = {'a': {'type': LiteralInput, 'height': 100}}
    assert len(model.children) == 3
    title, number, text = model.children
    assert isinstance(title, Div)
    assert isinstance(number, TextInput)
    assert number.height == 100
    assert isinstance(text, TextInput)