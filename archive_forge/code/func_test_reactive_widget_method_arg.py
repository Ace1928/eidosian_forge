import numpy as np
import pandas as pd
import param
from param import rx
from panel.layout import Row, WidgetBox
from panel.pane.base import PaneBase
from panel.param import ReactiveExpr
from panel.widgets import IntSlider
def test_reactive_widget_method_arg():
    slider = IntSlider()
    expr = ReactiveExpr(rx('{}').format(slider))
    assert slider in expr.widgets