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
def test_rerender_bounded_widget_when_bounds_set_and_unset():

    class Test(param.Parameterized):
        num = param.Range()
    test = Test()
    p = Param(test)
    assert isinstance(p._widgets['num'], LiteralInput)
    assert p._widgets['num'] in p._widget_box
    test.param.num.bounds = (0, 5)
    assert isinstance(p._widgets['num'], RangeSlider)
    assert p._widgets['num'] in p._widget_box
    test.param.num.bounds = (None, 5)
    assert isinstance(p._widgets['num'], LiteralInput)
    assert p._widgets['num'] in p._widget_box