from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_string_list_interact():

    def test(a):
        return a
    options = ['A', 'B', 'C']
    interact_pane = interactive(test, a=options)
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.Select)
    assert widget.value == 'A'
    assert widget.options == options