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
def test_single_param(document, comm):

    class Test(param.Parameterized):
        a = param.Parameter(default=0)
    test = Test()
    test_pane = Param(test.param.a)
    model = test_pane.get_root(document, comm=comm)
    assert isinstance(model, BkColumn)
    assert len(model.children) == 1
    widget = model.children[0]
    assert isinstance(widget, TextInput)
    assert widget.value == '0'