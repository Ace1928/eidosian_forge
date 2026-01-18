import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_setitem(document, comm):
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    p1, p2 = tabs.objects
    model = tabs.get_root(document, comm=comm)
    tab1, tab2 = model.tabs
    assert p1._models[model.ref['id']][0] is tab1.child
    div3 = Div()
    tabs[0] = ('C', div3)
    tab1, tab2 = model.tabs
    assert tab1.child is div3
    assert tab1.title == 'C'
    assert tab2.child is div2
    assert p1._models == {}