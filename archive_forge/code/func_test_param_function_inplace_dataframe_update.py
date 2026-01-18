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
def test_param_function_inplace_dataframe_update(document, comm):
    number = NumberInput(value=0)

    def layout(value):
        return pd.DataFrame({'x': [0, 1, 2], 'y': [0, 1, value]})
    pane = ParamFunction(bind(layout, number), inplace=True)
    root = pane.get_root(document, comm)
    model = root.children[0]
    html_table = model.text
    assert html_table.startswith('&lt;table class=&quot;dataframe panel-df&quot;&gt;\n  &lt;thead&gt;\n')
    number.value = 314
    assert model is root.children[0]
    assert model.text != html_table
    assert '314' in model.text
    html_table = model.text
    number.param.trigger('value')
    assert model.text is html_table