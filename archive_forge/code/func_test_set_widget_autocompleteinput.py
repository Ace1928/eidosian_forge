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
def test_set_widget_autocompleteinput(document, comm):

    class Test(param.Parameterized):
        choice = param.Selector(default='', objects=['a', 'b'], check_on_set=False)
    test = Test()
    test_pane = Param(test, widgets={'choice': AutocompleteInput})
    model = test_pane.get_root(document, comm=comm)
    autocompleteinput = model.children[1]
    assert isinstance(autocompleteinput, BkAutocompleteInput)
    if Version(param.__version__) > Version('2.0.0a2'):
        assert autocompleteinput.completions == ['a', 'b', '']
    else:
        assert autocompleteinput.completions == ['a', 'b']
    assert autocompleteinput.value == ''
    assert autocompleteinput.disabled == False
    test.choice = 'b'
    assert autocompleteinput.value == 'b'
    test.param['choice'].objects = ['c', 'd']
    assert autocompleteinput.completions == ['c', 'd']
    assert autocompleteinput.value == ''