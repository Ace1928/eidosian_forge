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
def test_param_subobject_expand_no_toggle(document, comm):

    class Test(param.Parameterized):
        a = param.Parameter()
    test = Test(a=Test(name='Nested'))
    test_pane = Param(test, expand=True, expand_button=False)
    model = test_pane.get_root(document, comm=comm)
    assert len(model.children) == 3
    _, _, subpanel = test_pane.layout.objects
    div, widget = model.children[2].children
    assert div.text == '<b>Nested</b>'
    assert isinstance(widget, BkTextInput)