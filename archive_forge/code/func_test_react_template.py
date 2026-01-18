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
def test_react_template(document, comm):
    tmplt = ReactTemplate(title='BasicTemplate', header_background='blue', header_color='red')
    tmplt._update_vars()
    tvars = tmplt._render_variables
    assert tvars['app_title'] == 'BasicTemplate'
    assert tvars['header_background'] == 'blue'
    assert tvars['header_color'] == 'red'
    assert tvars['nav'] == False
    assert tvars['busy'] == True
    assert tvars['header'] == False
    assert tvars['rowHeight'] == tmplt.row_height
    assert tvars['breakpoints'] == tmplt.breakpoints
    assert tvars['cols'] == tmplt.cols
    markdown = Markdown('# Some title')
    tmplt.main[:4, :6] = markdown
    markdown2 = Markdown('# Some title')
    tmplt.main[:4, 6:] = markdown2
    layouts = {'lg': [{'h': 4, 'i': '1', 'w': 6, 'x': 0, 'y': 0}, {'h': 4, 'i': '2', 'w': 6, 'x': 6, 'y': 0}], 'md': [{'h': 4, 'i': '1', 'w': 6, 'x': 0, 'y': 0}, {'h': 4, 'i': '2', 'w': 6, 'x': 6, 'y': 0}]}
    for size in layouts:
        for layout in layouts[size]:
            layout.update({'minW': 0, 'minH': 0})
    assert json.loads(tvars['layouts']) == layouts