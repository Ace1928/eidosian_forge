import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_set_panes(document, comm):
    div1, div2 = (Div(), Div())
    p1 = pn.panel(div1, name='Div1')
    p2 = pn.panel(div2, name='Div2')
    tabs = Tabs(p1, p2)
    model = tabs.get_root(document, comm=comm)
    div3 = Div()
    p3 = pn.panel(div3, name='Div3')
    tabs.objects = [p1, p2, p3]
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 3
    assert all((isinstance(c, BkPanel) for c in model.tabs))
    tab1, tab2, tab3 = model.tabs
    assert tab1.title == tab1.name == p1.name == 'Div1'
    assert tab1.child is div1
    assert tab2.title == tab2.name == p2.name == 'Div2'
    assert tab2.child is div2
    assert tab3.title == tab3.name == p3.name == 'Div3'
    assert tab3.child is div3