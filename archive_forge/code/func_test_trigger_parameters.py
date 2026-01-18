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
def test_trigger_parameters(document, comm):

    class Test(param.Parameterized):
        a = param.ListSelector(objects=[1, 2, 3, 4], default=list())
    t = Test()
    t.a.append(4)
    pane = Param(t.param.a)
    t.a.append(1)
    t.param.trigger('a')
    assert pane[0].value == [4, 1]