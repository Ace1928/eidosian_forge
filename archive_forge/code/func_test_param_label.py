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
def test_param_label(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5), label='A')
        b = param.Action(label='B')
    test = Test()
    test_pane = Param(test)
    a_param = test.param['a']
    a_param.label = 'B'
    assert test_pane._widgets['a'].name == 'B'
    b_param = test.param['b']
    b_param.label = 'C'
    assert test_pane._widgets['b'].name == 'C'