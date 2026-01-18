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
def test_reactive_servable_title():
    doc = Document()
    session_context = unittest.mock.Mock()
    with patch_curdoc(doc):
        doc._session_context = lambda: session_context
        ReactiveHTML().servable(title='A')
        ReactiveHTML().servable(title='B')
    assert doc.title == 'B'