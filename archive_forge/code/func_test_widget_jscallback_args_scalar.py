import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_jscallback_args_scalar(document, comm):
    widget = ColorPicker(value='#ff00ff')
    widget.jscallback(value='some_code', args={'scalar': 1})
    model = widget.get_root(document, comm=comm)
    customjs = model.js_property_callbacks['change:color'][-1]
    assert customjs.args['scalar'] == 1