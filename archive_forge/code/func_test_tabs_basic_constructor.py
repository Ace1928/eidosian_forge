import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_basic_constructor(document, comm):
    tabs = Tabs('plain', 'text')
    model = tabs.get_root(document, comm=comm)
    assert isinstance(model, BkTabs)
    assert len(model.tabs) == 2
    assert all((isinstance(c, BkPanel) for c in model.tabs))
    tab1, tab2 = model.tabs
    assert 'plain' in tab1.child.text
    assert 'text' in tab2.child.text