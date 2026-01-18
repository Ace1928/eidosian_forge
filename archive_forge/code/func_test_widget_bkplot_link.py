import pytest
from bokeh.plotting import figure
from panel.layout import Row
from panel.links import Link
from panel.pane import Bokeh, HoloViews
from panel.tests.util import hv_available
from panel.widgets import (
def test_widget_bkplot_link(document, comm):
    widget = ColorPicker(value='#ff00ff')
    bokeh_fig = figure()
    scatter = bokeh_fig.scatter([1, 2, 3], [1, 2, 3])
    widget.jslink(scatter.glyph, value='fill_color')
    row = Row(bokeh_fig, widget)
    model = row.get_root(document, comm=comm)
    link_customjs = model.children[1].js_property_callbacks['change:color'][-1]
    assert link_customjs.args['source'] is model.children[1]
    assert link_customjs.args['target'] is scatter.glyph
    assert scatter.glyph.fill_color == '#ff00ff'
    code = "\n    var value = source['color'];\n    value = value;\n    value = value;\n    try {\n      var property = target.properties['fill_color'];\n      if (property !== undefined) { property.validate(value); }\n    } catch(err) {\n      console.log('WARNING: Could not set fill_color on target, raised error: ' + err);\n      return;\n    }\n    try {\n      target['fill_color'] = value;\n    } catch(err) {\n      console.log(err)\n    }\n    "
    assert link_customjs.code == code