from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_tuple_range_interact_with_no_step_and_value():

    def test(a):
        return a
    interact_pane = interactive(test, a=(0, 4, None, 0))
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.IntSlider)
    assert widget.value == 0
    assert widget.start == 0
    assert widget.end == 4