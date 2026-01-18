import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_transform_dependency():
    widget = IntSlider()
    assert transform_reference(widget) is widget.param.value