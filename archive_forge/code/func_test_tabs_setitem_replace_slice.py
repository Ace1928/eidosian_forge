import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_setitem_replace_slice(document, comm):
    div1 = Div()
    div2 = Div()
    div3 = Div()
    layout = Tabs(('A', div1), ('B', div2), ('C', div3))
    p1, p2, p3 = layout.objects
    model = layout.get_root(document, comm=comm)
    assert p1._models[model.ref['id']][0] is model.tabs[0].child
    div3 = Div()
    div4 = Div()
    layout[1:] = [('D', div3), ('E', div4)]
    tab1, tab2, tab3 = model.tabs
    assert tab1.child is div1
    assert tab1.title == 'A'
    assert tab2.child is div3
    assert tab2.title == 'D'
    assert tab3.child is div4
    assert tab3.title == 'E'
    assert p2._models == {}
    assert p3._models == {}