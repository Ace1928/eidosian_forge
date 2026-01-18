import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_pop(document, comm):
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    p1, p2 = tabs.objects
    model = tabs.get_root(document, comm=comm)
    tab1 = model.tabs[0]
    assert p1._models[model.ref['id']][0] is tab1.child
    tabs.pop(0)
    assert len(model.tabs) == 1
    tab1 = model.tabs[0]
    assert tab1.child is div2
    assert p1._models == {}