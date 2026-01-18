import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_radd_list(document, comm):
    div1 = Div()
    div2 = Div()
    tabs1 = Tabs(('Div1', div1), ('Div2', div2))
    div3 = Div()
    div4 = Div()
    combined = [('Div3', div3), ('Div4', div4)] + tabs1
    model = combined.get_root(document, comm=comm)
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 4
    assert all((isinstance(c, BkPanel) for c in model.tabs))
    tab3, tab4, tab1, tab2 = model.tabs
    assert tab1.title == 'Div1'
    assert tab1.child is div1
    assert tab2.title == 'Div2'
    assert tab2.child is div2
    assert tab3.title == 'Div3'
    assert tab3.child is div3
    assert tab4.title == 'Div4'
    assert tab4.child is div4