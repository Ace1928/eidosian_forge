import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_setitem_replace_slice_error(document, comm):
    div1 = Div()
    div2 = Div()
    div3 = Div()
    layout = Tabs(div1, div2, div3)
    layout.get_root(document, comm=comm)
    div3 = Div()
    with pytest.raises(IndexError):
        layout[1:] = [div3]