from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_string_interact():

    def test(a):
        return a
    interact_pane = interactive(test, a='')
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.TextInput)
    assert widget.value == ''