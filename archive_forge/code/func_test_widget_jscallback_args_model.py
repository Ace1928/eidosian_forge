import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_jscallback_args_model(document, comm):
    widget = ColorPicker(value='#ff00ff')
    widget2 = ColorPicker(value='#ff00ff')
    widget.jscallback(value='some_code', args={'widget': widget2})
    model = Row(widget, widget2).get_root(document, comm=comm)
    customjs = model.children[0].js_property_callbacks['change:color'][-1]
    assert customjs.args['source'] is model.children[0]
    assert customjs.args['widget'] is model.children[1]
    assert customjs.code == 'try { some_code } catch(err) { console.log(err) }'