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
def test_param_function_recursive_update(document, comm):
    checkbox = Checkbox(value=False)

    def layout(value):
        return Row(Markdown(f'{value}'))
    pane = ParamFunction(bind(layout, checkbox), inplace=True)
    root = pane.get_root(document, comm)
    layout = root.children[0]
    assert layout.children[0].text == '&lt;p&gt;False&lt;/p&gt;\n'
    checkbox.value = True
    assert layout is root.children[0]
    assert layout.children[0].text == '&lt;p&gt;True&lt;/p&gt;\n'