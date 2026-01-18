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
def test_reactive_html_no_id_method_error():
    with pytest.raises(ValueError) as excinfo:

        class Test(ReactiveHTML):
            _template = '<div onclick=${_onclick}></div>'

            def _onclick(self):
                pass
    assert 'Found <div> node with the `onclick` callback referencing the `_onclick` method.' in str(excinfo.value)