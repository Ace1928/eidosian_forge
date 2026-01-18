import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_layout_default_with_widgets():
    w = IntSlider(value=2, start=1, end=5)
    i = rx(1)
    layout = ReactiveExpr(i + w).layout
    assert isinstance(layout, Row)
    assert len(layout) == 1
    assert isinstance(layout[0], Row)
    assert len(layout[0]) == 2
    assert isinstance(layout[0][0], WidgetBox)
    assert isinstance(layout[0][1], PaneBase)
    assert len(layout[0][0]) == 1
    assert isinstance(layout[0][0][0], IntSlider)