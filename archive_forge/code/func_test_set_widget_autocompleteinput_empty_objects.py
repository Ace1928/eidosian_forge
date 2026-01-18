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
def test_set_widget_autocompleteinput_empty_objects(document, comm):

    class Test(param.Parameterized):
        choice = param.Selector(default='', objects=[], check_on_set=False)
    test = Test()
    test_pane = Param(test, widgets={'choice': AutocompleteInput})
    model = test_pane.get_root(document, comm=comm)
    autocompleteinput = model.children[1]
    assert isinstance(autocompleteinput, BkAutocompleteInput)
    assert autocompleteinput.completions == ['']
    assert autocompleteinput.value == ''
    assert autocompleteinput.disabled == False