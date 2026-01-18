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
def test_param_throttled(document, comm):

    class Test(param.Parameterized):
        a = param.Number(default=1.2, bounds=(0, 5), label='A')
        b = param.Number(default=1.2, bounds=(0, 5), label='B')
    test = Test()
    try:
        param.parameterized.warnings_as_exceptions = True
        test_pane = Param(test, widgets={'b': {'throttled': True}})
    finally:
        param.parameterized.warnings_as_exceptions = False
    model = test_pane.get_root(document, comm=comm)
    assert len(model.children) == 3
    _, ma, mb = model.children
    ma.value = 1
    assert ma.value == 1
    assert ma.value_throttled == 1.2
    assert test.a == 1
    test.a = 2
    assert ma.value == 2
    assert ma.value_throttled == 1.2
    assert test.a == 2
    test_pane._widgets['b']._process_events({'value_throttled': 3})
    assert mb.value != 3
    assert test.b == 3
    test_pane._widgets['b']._process_events({'value': 4})
    assert test.b == 3
    assert mb.value == 4
    test.b = 5
    assert mb.value == 5
    assert test_pane._widgets['b'].value == 5
    assert test_pane._widgets['b'].value_throttled == 5