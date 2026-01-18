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
def test_template_server_title():
    tmpl = VanillaTemplate(title='Main title')
    doc = Document()
    with patch_curdoc(doc):
        doc = tmpl.server_doc(title='Ignored title')
    assert doc.title == 'Main title'