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
def test_number_param_overrides(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=0.1, bounds=(0, 1.1))
    test = Test()
    test_pane = Param(test, widgets={'a': {'value': 0.3, 'start': 0.1, 'end': 1.0}})
    model = test_pane.get_root(document, comm=comm)
    widget = model.children[1]
    assert isinstance(widget, Slider)
    assert widget.start == 0.1
    assert widget.end == 1.0
    assert widget.value == 0.3