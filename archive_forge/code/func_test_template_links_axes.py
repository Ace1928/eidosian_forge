import json
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from panel.layout import GridSpec, Row
from panel.pane import HoloViews, Markdown
from panel.template import (
from panel.template.base import BasicTemplate
from panel.widgets import FloatSlider
from .util import hv_available
@hv_available
def test_template_links_axes(document, comm):
    tmplt = Template(template)
    p1 = HoloViews(hv.Curve([1, 2, 3]), backend='bokeh')
    p2 = HoloViews(hv.Curve([1, 2, 3]), backend='bokeh')
    p3 = HoloViews(hv.Curve([1, 2, 3]), backend='bokeh')
    row = Row(p2, p3)
    tmplt.add_panel('A', p1)
    tmplt.add_panel('B', row)
    tmplt._init_doc(document, comm, notebook=True)
    _, (m1, _) = list(p1._models.items())[0]
    _, (m2, _) = list(p2._models.items())[0]
    _, (m3, _) = list(p3._models.items())[0]
    assert m1.x_range is m2.x_range
    assert m1.y_range is m2.y_range
    assert m2.x_range is m3.x_range
    assert m2.y_range is m3.y_range