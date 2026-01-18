from __future__ import annotations
import asyncio
import unittest.mock
from functools import partial
from typing import ClassVar, Mapping
import bokeh.core.properties as bp
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from bokeh.models import Div
from panel.depends import bind, depends
from panel.layout import Tabs, WidgetBox
from panel.pane import Markdown
from panel.reactive import Reactive, ReactiveHTML
from panel.viewable import Viewable
from panel.widgets import (
def test_reactive_html_templated_literal_add_loop_id_and_for_loop_var_insensitive_to_spaces():

    class TestTemplatedChildren(ReactiveHTML):
        children = param.List(default=[])
        _template = '\n        <select id="select">\n        {%- for option in children %}\n          <option id="option">{{option   }}</option>\n        {%- endfor %}\n        </select>\n        '
    assert TestTemplatedChildren._node_callbacks == {}
    assert TestTemplatedChildren._inline_callbacks == []
    assert TestTemplatedChildren._parser.children == {}
    test = TestTemplatedChildren(children=['A', 'B', 'C'])
    assert test._get_template()[0] == '\n        <select id="select-${id}">\n          <option id="option-0-${id}">A</option>\n          <option id="option-1-${id}">B</option>\n          <option id="option-2-${id}">C</option>\n        </select>\n        '
    model = test.get_root()
    assert test._attrs == {}
    assert model.looped == ['option']