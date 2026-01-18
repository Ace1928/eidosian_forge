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
def test_number_input_none_support():

    class Test(param.Parameterized):
        number = param.Number(default=0, allow_None=True)
        none = param.Number(default=None, allow_None=True)
    test_widget = Param(Test())
    assert test_widget[1].value == 0
    assert test_widget[2].value is None