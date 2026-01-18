import random
from panel import (
from panel.pane import HTML
from panel.widgets import IntSlider, Trend
def test_trend_auto_value(document, comm):
    data = {'x': [1, 2, 3, 4, 5], 'y': [3800, 3700, 3800, 3900, 4000]}
    trend = Trend(data=data)
    model = trend.get_root(document, comm)
    assert model.value == 4000
    assert model.value_change == 4000 / 3900 - 1