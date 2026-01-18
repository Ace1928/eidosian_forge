import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_close_tab_on_server(document, comm, tabs):
    model = tabs.get_root(document, comm=comm)
    _, div2 = tabs
    tabs._server_change(document, model.ref['id'], None, 'tabs', model.tabs, model.tabs[1:])
    assert len(tabs.objects) == 1
    assert tabs.objects[0] is div2