import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_tabs_clone_args():
    div1 = Div()
    div2 = Div()
    tabs = Tabs(div1, div2)
    clone = tabs.clone(div2, div1)
    assert tabs.objects[0].object is clone.objects[1].object
    assert tabs.objects[1].object is clone.objects[0].object