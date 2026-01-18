import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_pane_update(document, comm):
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    model = tabs.get_root(document, comm=comm)
    new_div = Div()
    tabs[1].object = new_div
    assert model.tabs[1].child is new_div