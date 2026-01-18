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
def test_set_show_name(document, comm):

    class Test(param.Parameterized):
        a = param.Number(bounds=(0, 10))
    pane = Param(Test())
    model = pane.get_root(document, comm=comm)
    assert len(model.children) == 2
    title, widget = model.children
    assert isinstance(title, Div)
    assert isinstance(widget, Slider)
    pane.show_name = False
    assert len(model.children) == 1
    assert isinstance(model.children[0], Slider)