import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_clear(document, comm):
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    p1, p2 = tabs.objects
    model = tabs.get_root(document, comm=comm)
    tabs.clear()
    assert tabs._names == []
    assert len(model.tabs) == 0
    assert p1._models == p2._models == {}