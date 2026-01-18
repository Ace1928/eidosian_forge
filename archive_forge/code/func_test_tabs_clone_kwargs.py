import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_clone_kwargs():
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    clone = tabs.clone(width=400, sizing_mode='stretch_height')
    assert clone.width == 400
    assert clone.sizing_mode == 'stretch_height'