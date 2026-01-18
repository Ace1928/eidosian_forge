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
def test_range_param(document, comm):

    class Test(param.Parameterized):
        a = param.Range(default=(0.1, 0.5), bounds=(0, 1.1))
    test = Test()
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    widget = model.children[1]
    assert isinstance(widget, BkRangeSlider)
    assert widget.start == 0
    assert widget.end == 1.1
    assert widget.value == (0.1, 0.5)
    test.a = (0.2, 0.4)
    assert widget.value == (0.2, 0.4)
    a_param = test.param['a']
    a_param.bounds = (0.1, 0.6)
    assert widget.start == 0.1
    assert widget.end == 0.6
    a_param.constant = True
    assert widget.disabled == True
    test_pane._cleanup(model)
    a_param.constant = False
    a_param.bounds = (-1, 1)
    test.a = (0.05, 0.2)
    assert widget.value == (0.2, 0.4)
    assert widget.start == 0.1
    assert widget.end == 0.6
    assert widget.disabled == True