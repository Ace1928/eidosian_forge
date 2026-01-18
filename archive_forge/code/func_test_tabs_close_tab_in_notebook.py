import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_close_tab_in_notebook(document, comm, tabs):
    model = tabs.get_root(document, comm=comm)
    old_tabs = list(model.tabs)
    _, div2 = tabs
    tabs._comm_change(document, model.ref['id'], comm, None, 'tabs', old_tabs, [model.tabs[1]])
    assert len(tabs.objects) == 1
    assert tabs.objects[0] is div2