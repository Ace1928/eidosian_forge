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
def test_object_selector_param_overrides(document, comm):

    class Test(param.Parameterized):
        a = param.ObjectSelector(default='b', objects=[1, 'b', 'c'])
    test = Test()
    test_pane = Param(test, widgets={'a': {'options': ['b', 'c'], 'value': 'c'}})
    model = test_pane.get_root(document, comm=comm)
    select = model.children[1]
    assert isinstance(select, Select)
    assert select.options == ['b', 'c']
    assert select.value == 'c'
    assert select.disabled == False