from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_tuple_range_interact_with_default_of_different_type():

    def test(a=3.1):
        return a
    interact_pane = interactive(test, a=(0, 4))
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.FloatSlider)
    assert widget.value == 3.1
    assert widget.start == 0
    assert widget.end == 4