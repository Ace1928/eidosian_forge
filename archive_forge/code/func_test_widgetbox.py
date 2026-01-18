import param
import pytest
from bokeh.models import Column as BkColumn, Div, Row as BkRow
from panel.chat import ChatInterface
from panel.layout import (
from panel.layout.base import ListPanel, NamedListPanel
from panel.pane import Bokeh, Markdown
from panel.param import Param
from panel.tests.util import check_layoutable_properties
from panel.widgets import Debugger, MultiSelect
def test_widgetbox(document, comm):
    widget_box = WidgetBox('WidgetBox')
    model = widget_box.get_root(document, comm=comm)
    assert isinstance(model, widget_box._bokeh_model)
    assert not widget_box.horizontal
    widget_box.horizontal = True
    assert widget_box.horizontal